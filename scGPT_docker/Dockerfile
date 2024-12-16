# Use the official PyTorch image with CUDA 12.1 support
#FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-devel
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Set the DEBIAN_FRONTEND to noninteractive to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    gnupg2 \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add Conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Clone the Git repository
RUN git clone https://github.com/zifornd/scGPT-multiomics.git /app/repo

# Set the working directory to the cloned repository
WORKDIR /app/repo

# Copy the requirements.txt file
COPY requirements.txt .

# Create the Conda environment
RUN conda env create -f requirements.txt -n myenv

# Activate the Conda environment
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Install dependencies using poetry
RUN poetry install

RUN pip install gputil anndata scanpy scvi-tools wandb torch==2.1.2 torchtext==0.16.2
RUN pip install scib muon seaborn mofax mofapy2 louvain
# Copy the rest of your application code into the container
COPY . .

# Command to run your application
ENTRYPOINT ["/bin/bash"]
#CMD ["python", "your_main_script.py"]
