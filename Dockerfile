# use python 3.9
FROM python:3.9-slim

# add name to docker
WORKDIR /app

# app equipments
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# list librabry to install in docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy files to docker
COPY . .

# docker port
EXPOSE 8501

# for runing app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]