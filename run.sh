#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <input_nifti_file.nii.gz> or <folder_path_with_nifti_files>"
    exit 1
fi

# Check if the input file exists and has .nii.gz extension
if [[ ( ! -f "$1" || ( "$1" != *.nii.gz && "$1" != *.nii ) ) && ! -d "$1" ]]; then
    echo "Error: Input file does not exist or is not a .nii.gz file"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating new Python virtual environment..."
    python3 -m venv venv
    
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Using existing virtual environment..."
    source venv/bin/activate
fi

# Make sure grace.py is executable
chmod +x grace.py

# Run the grace.py script with the input file
python ./grace.py "$1"

# Deactivate virtual environment
deactivate 