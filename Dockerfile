# Faeyon Training Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Faeyon (assuming it's in the current directory)
COPY . /app/faeyon
RUN pip install -e /app/faeyon

# Copy training scripts
COPY *.py /app/
COPY data/ /app/data/

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for distributed training
EXPOSE 29500

# Default command
CMD ["python", "training.py"]
