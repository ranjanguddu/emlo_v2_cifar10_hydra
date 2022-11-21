FROM python:3.9.13-slim

# Basic setup
RUN apt update
RUN apt install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

# Set working directory
WORKDIR /workspace/project

# Install requirements
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

#ENTRYPOINT [ "python3","src/demo_scripted.py" ]

COPY . .


# /workspace/project/logs/train/runs/2022-11-18_12-20-35
# /workspace/emlo_v2_cifar10_hydra/logs/train/runs/2022-11-18_12-20-35/model_script.pt