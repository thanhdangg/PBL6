FROM python:3.10
RUN apt-get update && apt-get install -y libgl1-mesa-glx
WORKDIR /usr/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 80
CMD ["python","main.py"]
