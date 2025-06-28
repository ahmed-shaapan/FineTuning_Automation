# Start with a modern NVIDIA CUDA base image compatible with your libraries
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set non-interactive environment for package installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker's layer caching
COPY requirements.txt .

# Install PyTorch with the correct CUDA version first. This is a best practice.
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
# Install the rest of your Python dependencies
RUN pip install -r requirements.txt

# Copy all of your project code into the container's /app/ directory
COPY . .

# This is the command RunPod will run to start your worker.
# It executes your handler script.
CMD ["python", "runpod_handler.py"]