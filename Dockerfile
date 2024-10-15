FROM python:3.10
RUN apt-get update && apt-get install -y libgl1-mesa-glx
WORKDIR /mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 3100
CMD ["python","main.py"]