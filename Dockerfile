FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential cmake ffmpeg

# Install pybind11 and numpy (needed by py-gpmf-parser)
RUN pip install pybind11 numpy

# Clone and install py-gpmf-parser
RUN git clone --recursive https://github.com/urbste/py-gpmf-parser.git
WORKDIR /py-gpmf-parser
RUN pip install .

# Copy the telemetry extraction script into the image
COPY extract.py /py-gpmf-parser/extract.py

# Set working directory for video mounts
WORKDIR /data

CMD ["python3", "/py-gpmf-parser/extract.py"]
