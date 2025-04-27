import numpy as np
import cv2
import os
import uuid
import zipfile
import tempfile
import json
from PIL import Image, ImageFilter, ImageOps
from typing import Union, NoReturn, Optional, Tuple
from numpy.typing import NDArray
from scipy.ndimage import label
from flask import Flask, request, jsonify, send_file, send_from_directory, after_this_request, Response
from flask_cors import CORS
from datetime import datetime
app = Flask(__name__)
CORS(app)
# CORS(app, origins=['http://0.0.0.0:4200']) 

UPLOAD_FOLDER = os.path.join(os.getcwd(), './users')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# THIS FUNCTION IS USED FOR COLOUR QUANTIZATION -> reducing num colours in an image
#see https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
def get_k_means(image_path: str, num_colors : int =10) -> Tuple[NDArray[np.uint8],NDArray[np.int32],NDArray[np.uint8]]:
    #load image and covert to RGB for easier processing   
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #resize image for easier processing - shape returns tuple of (rows, columns and channels) channels is if if image is colour
    # the [:2] returns the height and width, where :2 means it only takes the first 2 values (height and width)
    h, w = image.shape[:2] 
    #resize image to desired width/height
    ###image_resized = cv2.resize(image, (w // 1, h // 1))
    image_resized=image
    #reshape for clustering 
    # reshape is a numpy operation that turns it into a 2D array called pixels where each row is RGB pixel, shape is (num_pixels,3)
    # the -1 means we're using all the pixels in one long list where each row is a pixel, amnd each columns is one of the RGB values. 
    #this is so each row becomes a single pixels RGB colour value (R,G,B) 
    pixels = image_resized.reshape((-1, 3))
    # print("pixel array is",pixels)
    #apply clustering 
    #np.float is so openCV can process array of pixels - this array is x rows/samples with feature (feature is the rgb)
    #num_colours is number of clusters/colours we want at the end
    #none=no prexisting labels / colours assigned
    #stopping criteria is if either max iterations=100 or error margin=0.2 
    #10 is number of times it will run
    #chose random intial colours as centers
    #output of kmeans is  a touple (retval,labels,centers) retval is compactness score (distance from point to cluster center), labels is index of closest cluster center to pixel, and center is the rgb value of the centers
    kmeans = cv2.kmeans(np.float32(pixels), num_colors, None,
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.8),
                        10, cv2.KMEANS_RANDOM_CENTERS)
    #cluster centers
    #centers=kmeans[2] because its the 3rd value of the touple (retval [0],label[1],centers[2])
    #where centers is final colours found after clustering - standard kmeans output.
    #need to convert to int bc output is a float32
    #centers stores an array where each entry is the [R,G,B] values that are the cluster centers ie [[R,G,B],[R,G,B],[R,G,B]] OR [[colour1],[colour2],[colour3]]
    cluster_colours:NDArray[np.uint32] = np.uint8(kmeans[2]) 
    #assigns colour from center to each pixel - standard kmeans output.
    # 2D array - labels indicate which cluster the pixel belong to/what colour it should have
    #alt way to describe it is assigns pixel to one of the clusters colours using 
    #reshape labels back into 2D array of h and w
    #labels is an array of width X and height Y where each (X,Y)=K where K is the clusterCenter/Colour  ie if i have a 2x2 image with (red,pink),(blue,lightblue) then the image's pixels would look like [(1,1),(2,2)] where 1=red and 2=blue if my image is AxB pixels, labels would have AxB rows/columns of K values
    labels_array:NDArray[np.uint32] = kmeans[1].reshape(image_resized.shape[:2])  
    #map pixel to the nearest cluster colour
    # creates a new numpy array, where each val at position (x,y) = K or 
    #works using numpy's advanced indexing to created (height,width,K) array
    #replaces pixel in image with closest colour in centers - uses numpy fancy indexing, which maps the colours to the label's x/y position array
    new_image_array:NDArray[np.uint8] = cluster_colours[labels_array]
    #resize back to original size
    ###new_image_array_resized = cv2.resize(new_image_array, (w, h), interpolation=cv2.INTER_NEAREST)
    ###return(cluster_colours,labels_array,new_image_array_resized)
    return(cluster_colours,labels_array,new_image_array)

    
#this function should generate a black/white image called PBN_untagged_initial_image which is the processed image with the colours removed
def apply_canny(processed_image_array: NDArray[np.uint8]) -> str:
    #attempt to use canny function to reduce noise, and create edges
    t_lower: int=100     #lower Hysteresis Thresholding value -pixels below this are background
    t_upper: int=200     #upper Hysteresis Thresholding value  - pizels above this is objects
    aperture_size: int =7  #between 3-7 size of the sobel filter to calculate gradient in algo - higher num=more features
    L2grad: bool=True     #boolean indicates we want to use L2Gradient algorithm
    PBN_image_array: NDArray[np.uint8]=cv2.Canny(processed_image_array,t_lower,t_upper, apertureSize=aperture_size, L2gradient=L2grad)
    PBN_untagged_output: str="PBN_untagged_inital_image.png"
    # invert the colour of the image with edges - this is so it appears as white on black instead of black on white - uses PIL
    PBN=ImageOps.invert(Image.fromarray(PBN_image_array)).save(PBN_untagged_output)
    return PBN_untagged_output  

#this function is used to get the scaling factor of the image based on the array size and actual image size
def get_scaling_factor(image_array: NDArray[np.int8],image_path: str) -> Tuple[int,int]:
    img=Image.open(image_path)
    array_rows, array_cols = image_array.shape[:2] 
    image_width, image_height = img.size
    scaling_factor_x = image_width / array_cols
    scaling_factor_y = image_height / array_rows
   # print(f" scaling_factor_y: {scaling_factor_y}, width {scaling_factor_x}")
    return image_width, image_height

#this function is used to take the processed image's colours, find the centroids of each colour's cluster and then generate PBN_Processed_image 
def label_PBN(image_path: str, PBN_inital_image: str, image_array : NDArray[np.int8] ,cluster_colours: NDArray[np.uint8], labels_array: NDArray[np.int32]) -> str:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size=0.4
        font_color=(0,0,0)
        font_thickness=1
        #creates a new image from the inital image ()
        output_image_preformatted=cv2.imread(PBN_inital_image)
        if output_image_preformatted is None:
            raise ValueError(f"Failed to load {PBN_inital_image}")

        output_image: NDArray[np.uint8]=np.array(output_image_preformatted,dtype=np.uint8)
        # Initialize a list to store the coordinates of the cluster centers
        cluster_centers:List[Tuple[int, int, np.ndarray]] =[] 
        data: NDArray[np.int32]=labels_array
        image_width, image_height=get_scaling_factor(output_image, image_path)
        kernel=np.ones((image_width,image_height),np.uint8)
        #Iterate over each unique cluster 
        for colour in np.unique(labels_array):
            # Find all connected regions of the current cluster label
            # Label the connected components of the current cluster label (if any)
            #Binary Mask: For each cluster label, create a binary mask (True where the label matches the current cluster and False otherwise).
            binary_mask = (data == colour)  # Create a binary mask for the current cluster
            # mask=mask.astype(np.uint8)*255
            # binary_mask=cv2.dilate(mask,kernel,iterations=1)
            #se scipy.ndimage.label to identify distinct connected components (occurrences) of the same cluster label. Each connected component will be assigned a unique ID (starting from 1).
            labeled_array, num_features = label(binary_mask)  # Label connected regions (occurrences) #
           #For each connected region (identified by feature_id), find the coordinates of the pixels belonging to that region, then calculate the mean of these coordinates to find the center of the occurrence.
            for feature_id in range(1, num_features + 1):  # Starting from 1, as 0 is the background
                # Find coordinates of the current region's pixels
                coordinates = np.argwhere(labeled_array == feature_id)
                # Calculate the center of this occurrence by averaging the coordinates
                center = coordinates.mean(axis=0)  # Mean of rows and columns (center)
                # For each occurrence of a cluster, store the cluster label, occurrence ID, and center coordinates.
                cluster_centers.append((colour, feature_id, center))

        #Print the cluster centers (cluster label, occurrence id, and (row, col) center)
        for colour, feature_id, center in cluster_centers:
           y,x=np.array(center).astype(int) #used to debug array?
            #set the fond colour as the RGB value of the cluster
           font_colorWIP=cluster_colours[colour].tolist()
          # print(f"{x}, {y} colour {colour}, occurance {feature_id}") #used to debug
           #this code adds the text to the output image.
           cv2.putText(output_image, str(colour),(x*1,y*1), font, font_size, font_colorWIP, font_thickness) #**the x1 is if you do any resizing when taking the kmeans
            
    #   # Put the ID label at the center of the cluster
        PBN_output_path: str="PBN_processed_image.png"
        Image.fromarray(output_image).save(PBN_output_path)
        return PBN_output_path


def process_image(original_image:str ,num_colors: int, pre_medianblur_count : int =1 , pre_gaussian_blur_count: int =1 , post_medianblur_count: int=1 , post_gaussian_blur_count: int =1 ) -> NoReturn:
    #ensure only  0 or positive numbers are used to configure blurring.
    if pre_medianblur_count < 0:
        pre_medianblur_count = 0
    if pre_gaussian_blur_count < 0:
        pre_gaussian_blur_count = 0
    if post_medianblur_count < 0:
        post_medianblur_count = 0
    if post_gaussian_blur_count < 0:
        post_gaussian_blur_count = 0

    #CREATE INITAL CLUSTERED IMAGE and seperate the image in num_colour colours,   - see get kmeans function for more info on this
    print("starting initial k means")
    cluster_colours,labels_array,initial_image_array=get_k_means(original_image,num_colors)
    #ensures colours dont belnd during clustering by rounding to nearet values via blurring
    # initial_image_array=cv2.medianBlur(initial_image_array,9)
    # initial_image_array=cv2.GaussianBlur(initial_image_array,(7,7),0)
    # initial_image_array=cv2.medianBlur(initial_image_array,9)
    print("starting pre-processing")
    #these loops are used to perform pre-process blurring X many times to help reduce the amount of individual pixels
    for g1 in range(pre_gaussian_blur_count):
        initial_image_array=cv2.GaussianBlur(initial_image_array,(7,7),0)
    for m1 in range(pre_medianblur_count):
       initial_image_array=cv2.medianBlur(initial_image_array,9)
    #convert to PIL (pillow) image and save (turn array to image)   
    initial_image_path: str = "processed_image.png"
    Image.fromarray(initial_image_array).save(initial_image_path)
    #RERUN KMEANS CLUSTERING TO GET UPDATED LABELS ARRAYS (map colours to pixels) AFTER BLURRNG TO HELP FIGURE OUT HOW TO LABEL PBN COLOURS 
    print("starting 2nd processing")
    updated_cluster_colours,updated_labels_array,updated_image_array= get_k_means(initial_image_path,num_colors) 
    #these loops are used to perform a final post clustering/colour reduction blurring to help smoothe the 2nd image for PBN 
    for g2 in range(post_gaussian_blur_count):
        updated_image_array=cv2.GaussianBlur(updated_image_array,(7,7),0)
    for m2 in range(post_medianblur_count):
        updated_image_array=cv2.medianBlur(updated_image_array,9)
    #update processed image so it reflects the array that went through 2 sets of kmeans clustering and 2 sets of blurring  
    updated_image_path = "processed_image.png"
    Image.fromarray(updated_image_array).save(updated_image_path)
    colourNum = 1

    color_map_json = {
        "color_map": []
    }   

    print("generating colour map")
    for colourid in updated_cluster_colours:
        # print(f"{colourid} = {colourNum}", temp_file)
        color_map_json["color_map"].append({
            "color": colourid.tolist(),  # Ensure it's JSON serializable
            "label": colourNum
        })
        colourNum+=1
    with open("color_map.json.txt", "w") as json_file:
         json.dump(color_map_json, json_file, indent=4)
    #CREATE BOTH THE pbn inital untagged image AND PBN_TAGGED IMAGE using teh updated processed image
    print("creating B/W image")
    PBN_inital_output: str =apply_canny(updated_image_array)
    #uses the processed image, untagged PBN image and their paths colours and arrays to create a new labeled PBN image 
    print("labeling B/W image")
    PBN_output_path=label_PBN(updated_image_path,PBN_inital_output, updated_image_array, updated_cluster_colours, updated_labels_array)
   

def start() -> NoReturn :
    print("app starting")
    num_colors=10
    #the input numbers are how many times it should be blurred, default is 0
    process_image("test_image.png",num_colors,3,3,3,3)
    print("end of process")


@app.route('/get-pbn-image', methods=['POST'])
def get_PBN_image() -> Union[Response, tuple[Response, int]]:
    print("recieved /get-pbn-image request")
    print("Incoming request:", request)
    print("Files:", request.files)
    try:
        if 'input_image' not in request.files:
            return jsonify({"error": "No input_image uploaded"}), 400

        input_image = request.files['input_image']
        num_colors = int(request.form.get("num_colors", 10))
        premedian_blur_loop_count = int(request.form.get("premedian_blur_loop_count", 1))
        pregaussian_blur_loop_count = int(request.form.get("pregaussian_blur_loop_count", 1))
        postmedian_blur_loop_count = int(request.form.get("postmedian_blur_loop_count", 1))
        postgaussian_blur_loop_count = int(request.form.get("postgaussian_blur_loop_count", 1))
        print("recieved file:", request.files)
        if input_image.filename == '':
            return jsonify({"error": "No selected file"}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            input_image.save(temp_file)
            temp_image_path: str = temp_file.name
        if not os.path.exists(temp_image_path):
            raise FileNotFoundError(f"Temp image path doesn't exist: {temp_image_path}")

  
        #  # Generate unique session folder
        #     session_id = str(uuid.uuid4())
        #     if not os.path.exists(f"./users/{session_id}")
        #          os.makedirs(user_folder, exist_ok=True)
        
        process_image(
            temp_image_path,
            num_colors,
            premedian_blur_loop_count,
            pregaussian_blur_loop_count,
            postmedian_blur_loop_count,
            postgaussian_blur_loop_count
            )      
        #zip files
        zip_path="pbn_images.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write("color_map.json.txt")
            zipf.write("PBN_processed_image.png")
            zipf.write("PBN_untagged_inital_image.png")
            zipf.write("processed_image.png")
            zipf.write(temp_image_path,arcname=input_image.filename)
        # return zip when URL is called 
        @after_this_request
        def cleanup(response: Response) -> Response:
            try:
                os.remove(temp_image_path)
                os.remove(zip_path)
                os.remove("color_map.json.txt")
                print(f"Deleted temp files: {temp_image_path} and {zip_path}")
            except Exception as e:
                print(f"Error deleting temp files: {e}")
            return response

        response= send_file(zip_path, mimetype='application/zip', as_attachment=True, download_name="pbn_images.zip")
        return response
    except Exception as e:
       print(f"Error processing image: {e}")
       return jsonify({"error": str(e)}), 500

@app.route("/")
def hello_world() -> str:
    try: 
        return "<p>Hello from sheldon's paint by numbers app</p>"
    except Exception as e:
        return "could not open app" + e 

if __name__ == '__main__':
    print("PBN App starting at " + str(datetime.now()) ) 
    # app.run(host='0.0.0.0', port=5443, ssl_context=('/etc/ssl/certs/cert.pem', '/etc/ssl/certs/key.pem'), debug=True)  # Enable HTTPS
    app.run(host='0.0.0.0', port=5443, ssl_context=('./certs/cert.pem', './certs/key.pem'), debug=True)  # Enable HTTPS
    # app.run(host='0.0.0.0', port=5443, debug=True)
    print("PBN App ending at " + str(datetime.now()) ) 
    #for testing run the start() function
    #  start()
