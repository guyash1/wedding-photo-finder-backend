#!/bin/bash
# Install Python packages
pip install --no-cache-dir -r requirements.txt

# Force uninstall opencv-python and keep only opencv-python-headless
pip uninstall -y opencv-python opencv-contrib-python
pip install --no-cache-dir opencv-python-headless==4.8.1.78

