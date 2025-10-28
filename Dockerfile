# Use an official PyTorch CUDA runtime so torch + GPU "just works"
# (Change the tag if you need a different CUDA / torch combo)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Prevent Python from writing .pyc files & keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY domino.py /app/domino.py

# Install Python dependencies
RUN pip install numpy nibabel scipy monai einops

# Copy the rest of the application
COPY . /app

# Make sure domino.py is executable
RUN chmod +x /app/domino.py

# Use tini as PID 1 to handle signals cleanly
ENTRYPOINT ["python3", "./domino.py"]

# This lets users pass arguments directly:
# e.g. docker run --gpus all domino-gpu ./samples/brain.nii.gz