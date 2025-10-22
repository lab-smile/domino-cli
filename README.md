# DOMINO CLI

Domino CLI is a tool for processing NIfTI (.nii or .nii.gz) files using [domino model](https://github.com/lab-smile/domino), batch processing is also supported.

## Prerequisites

- Python 3.1x
- Ability to create virtual environments (`python3-venv`)
- Docker (Optional)


## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd domino-cli
```

2. Make sure the run script is executable:
```bash
chmod +x run.sh
```

3. Download `DOMINO.pth` file by filling out the following form to `domino-cli` directory.
```bash
https://github.com/lab-smile/DOMINO?tab=readme-ov-file#pre-trained-models
```

4. Change the defaults within `domino.py` file i.e. num_gpu or batch_size etc.

## Usage

### Using Local Installation

The tool can be run using the provided shell script:

```bash
./run.sh <input_nifti_file.nii.gz> or <folder_path_to_nifti_images>
```

For example:
```bash
./run.sh sample_image.nii.gz
./run.sh ./input_folder
```

### Using Docker

You can run the tool using Docker in two ways:

#### Using Docker directly:

1. Build the Docker image:
```bash
docker build -t domino-cli .
```

2. Run the container:
```bash
docker run -v $(pwd):/app domino-cli <input_nifti_file.nii.gz>
```

For example:
```bash
docker run -v $(pwd):/app domino-cli sample_image.nii.gz
```

#### Using Docker compose:
To run the repo with the following command, you need to change the command argument in the `docker-compose.yml` file. (For example: ['python', 'domino.py', 'input.nii'])
```bash
docker compose up --build
```


#### Using our published docker hub image
You can use the published docker hub image `nikmk26/domino-cli:latest`

```bash
docker run -v $(pwd):/app nikmk26/domino-cli:latest <input_nifti_file.nii.gz>
```


### What the script does:

1. Creates a Python virtual environment
2. Installs all required dependencies
3. Processes the input NIfTI file(s)
4. Outputs the results in the `outputs` folder in the current directory.

### Output

The processed files will be saved in the `outputs` directory with the following naming convention:
- `<input_filename>_pred_domino.nii(.gz)`: NIfTI format output

## Error Cases

The script will show an error message if:
- No input file is provided
- The input file doesn't exist
- The input file is not a .nii.gz or .nii file
- Input is neither a file nor a folder with NIfTI images

## Dependencies

All required Python packages are listed in `requirements.txt` and will be automatically installed in the virtual environment when running the script.

## Notes

- The script automatically handles the creation and cleanup of the Python virtual environment
- Each run creates a fresh virtual environment to ensure consistency
- GPU support is available if CUDA is properly configured on your system
- Change `python` command in `run.sh` if command installed on your machine is `python3.x`
- If you are running on hipergator make sure you have >=python3.10 loaded, you can load it using `module load python/3.10`
