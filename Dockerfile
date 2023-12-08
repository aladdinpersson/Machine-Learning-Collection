FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
RUN pip install --upgrade pip
WORKDIR /code
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt
