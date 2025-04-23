FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set workspace
WORKDIR /workspace

# Default user
USER vscode

CMD ["bash"]