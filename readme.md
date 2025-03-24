Requirements
#apt-get update
#apt-get install python3-pip
#apt-get install python3
#pip3 install numpy
#pip3 install opencv-python-headless
#pip3 install pillow 
#pip3 install flask
#pip3 install zipfile36

How to Run as flask app
 python3 ./paint_by_nums.py 
OR if using flask runner 
export FLASK_APP=your_script.py
flask run --host=0.0.0.0 --port=5000 --cert=cert.pem --key=key.pem



HOW TO debug
uncomment the start function in, and edit the inputs and run  python3 ./paint_by_nums.py

if __name__ == '__main__':
    #for testing run the start() function
    #  start()

HOW to access code
api will run on https://localhost:5000/ and https://localhost:5000/get-image


CERTS - generated  self signed cert.pem and key.pem using openssl
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes

CERTs - hosting
Use lets encrypt will have to change ssl_context to ssl_context=('/etc/letsencrypt/live/yourdomain.com/fullchain.pem', '/etc/letsencrypt/live/yourdomain.com/privkey.pem')

TODO: use a revers nginx proxy or something to handle certs?
TODO: add sessions??
TODO: front end tbd
