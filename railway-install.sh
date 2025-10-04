#!/bin/bash
# Install system dependencies
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgthread-2.0-0

# Install Python packages
pip install --no-cache-dir -r requirements.txt

