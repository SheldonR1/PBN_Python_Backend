FROM python:3.11-slim
WORKDIR /PBN-Workspace
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 5443
CMD ["python", "paint_by_nums.py"]
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
