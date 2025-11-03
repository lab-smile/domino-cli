#!/bin/bash

# Function to log messages with timestamp
log() {
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S,$(date +%N | cut -c1-3)")
    if [ "$1" == "-n" ]; then
        shift
        echo -n "$timestamp | INFO | $*"
        return
    fi
    echo "$timestamp | INFO | $*"
}

# Check if input file is provided
if [ $# -lt 1 ]; then
    timestamp=$(date +"%Y-%m-%d %H:%M:%S,%3N")
    echo "$timestamp | WARN | Usage: ./run.sh <input_nifti_file.nii.gz> or <folder_path_with_nifti_files> [additional_options]"
    echo "$timestamp | WARN | Atleast one argument (input file or folder) is required."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log "Creating new Python virtual environment..."
    log "Using venv folder for the virtual environment."
    python3 -m venv venv 
    
    log -n "Installing dependencies... "
    source venv/bin/activate
    START_TIME=$(date +%s)
    (pip install -r requirements.txt --quiet 2>/dev/null) &
    PID=$!
    spinner="/-\|"
    i=0
    while kill -0 $PID 2>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\b${spinner:$i:1}"
        sleep 0.1
    done
    wait $PID
    STATUS=$?

    printf "\b \b"
    if [ $STATUS -ne 0 ]; then
        echo ""
        timestamp=$(date +"%Y-%m-%d %H:%M:%S,%3N")
        echo "$timestamp | ERROR | Error installing dependencies. Exiting."
        deactivate
        exit 1
    fi
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    echo ""
    log "Dependencies installed in $ELAPSED_TIME seconds."

else
    log "Using existing virtual environment..."
    source venv/bin/activate
fi

# Make sure domino.py is executable
chmod +x domino.py

# Run the domino.py script with the input file
python ./domino.py $*

# Deactivate virtual environment
deactivate 