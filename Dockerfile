FROM pytorch/pytorch:latest
WORKDIR /code
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8888
