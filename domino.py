#!/usr/bin/env python3
import os
import sys
import json
import time
import torch
import logging
import numpy as np
import nibabel as nib
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.data import MetaTensor, DataLoader, Dataset, load_decathlon_datalist
from monai.transforms import Compose, Spacingd, Orientationd, EnsureTyped, LoadImaged, EnsureChannelFirstd, CropForegroundd, MapTransform, Resized

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

logger=logging.getLogger(__name__)


# HOW TO USE:
# logger.info("Created output directories.")

def send_progress(message, progress):
    """
        Helper function to print SSE progress updates
        @param message: Message about current stage of model prediction (str)
        @param progress: Progress percentage (int)
    """
    # data = json.dumps({"message": message, "progress": progress})
    # return f"data: {data}\n\n"
    if progress != ".":
        logger.info(f"{message}... {progress}%")
    else:
        logger.info(f"{message}...")

def generate_datalist(folder_path):
    nii_files = [
        {"image": os.path.abspath(os.path.join(folder_path, f))}
        for f in os.listdir(folder_path)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ]

    datalist = {"testing": nii_files}

    with open("datalist.json", "w") as f:
        json.dump(datalist, f, indent=4)

def load_model(model_path, spatial_size, num_classes, device):
    """
        Load and configure the model for inference.
        @param model_path: Path to the model weights file (str)
        @param spatial_size: Size of the input images (tuple)
        @param num_classes: Number of output classes (int)
        @param device: Device to run the model on (str or torch.device)
        @param dataparallel: Whether to use DataParallel (bool)
        @param num_gpu: Number of GPUs to use if dataparallel is True (int)
        @return: Configured model for inference (torch.nn.Module)
    """
    send_progress("Configuring model", 10)

    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=spatial_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        proj_type="perceptron",
    )

    # if dataparallel:
    #     yield send_progress("Initializing DataParallel with multiple GPUs", 15)
    #     model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    model = model.to(device)
    send_progress(f"Loading model weights from {model_path}", 20)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    send_progress("Model loaded successfully.", 40)
    return model

class ConditionalNormalizationd(MapTransform):
    def __init__(self, keys, a_min, a_max, complexity_threshold=10000):
        super().__init__(keys)
        self.a_min = a_min
        self.a_max = a_max
        self.threshold = complexity_threshold

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]

            # MetaTensor check
            if isinstance(image, MetaTensor):
                original_meta = image.meta
                image_np = image.as_tensor().cpu().numpy()  # Convert to NumPy for percentile ops
            else:
                original_meta = None
                image_np = image  # Assume NumPy

            mean_val = image_np.mean()

            if mean_val > self.threshold:
                pmin, pmax = np.percentile(image_np, [10, 90])
                image_np = np.clip(image_np, pmin, pmax)
                image_np = (image_np - pmin) / (pmax - pmin + 1e-8)
            else:
                image_np = np.clip(image_np, self.a_min, self.a_max)
                image_np = (image_np - self.a_min) / (self.a_max - self.a_min + 1e-8)

            image_np = image_np.astype(np.float32)

            # Re-wrap if originally MetaTensor
            if original_meta is not None:
                image = MetaTensor(image_np, meta=original_meta)
            else:
                image = image_np

            d[key] = image  # ✅ THIS IS CRITICAL
        return d


def preprocess_datalists(a_min, a_max, complexity_threshold=10000):
    return Compose([
        LoadImaged(keys=["image"]),
        ConditionalNormalizationd(keys=["image"], a_min=a_min, a_max=a_max, complexity_threshold=complexity_threshold),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="trilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], track_meta=True)
    ])

def preprocess_input(input_path, device, a_min_value, a_max_value, complexity_threshold=10000):
    """
        Load and preprocess the input NIfTI image.
        @param input_path: Path to the input NIfTI image file (str)
        @param device: Device to run the preprocessing on (str or torch.device)
        @param a_min_value: Minimum intensity value for scaling (int or float)
        @param a_max_value: Maximum intensity value for scaling (int or float)
    """
    def normalize_fixed(data, a_min, a_max):
        data = np.clip(data, a_min, a_max)
        return (data - a_min) / (a_max - a_min + 1e-8)

    def normalize_percentile(data, lower=10, upper=90):
        pmin, pmax = np.percentile(data, [lower, upper])
        data = np.clip(data, pmin, pmax)
        return (data - pmin) / (pmax - pmin + 1e-8)
    
    send_progress(f"Loading input image from {input_path}", 30)
    input_img = nib.load(input_path)
    image_data = input_img.get_fdata().astype(np.float32)
    send_progress(f"Input image loaded. Shape: {image_data.shape}", 35)
    # image_max = np.max(image_data)
    # image_min = np.min(image_data)
    image_mean = np.mean(image_data)

    if image_mean > complexity_threshold:
        image_data = normalize_percentile(image_data)
        send_progress(f"Applied percentile normalization (due to mean > {complexity_threshold})", ".")
    else:
        image_data = normalize_fixed(image_data, a_min_value, a_max_value)


    # Convert to MetaTensor for MONAI compatibility
    meta_tensor = MetaTensor(image_data, affine=input_img.affine)

    send_progress("Applying preprocessing transforms", 40)
    
    # Apply MONAI test transforms
    test_transforms = Compose(
        [
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("trilinear"),
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )

    data = {"image": meta_tensor}
    transformed_data = test_transforms(data)

    # Convert to PyTorch tensor
    image_tensor = transformed_data["image"].clone().detach().unsqueeze(0).unsqueeze(0).to(device)
    send_progress(f"Preprocessing complete. Model input shape: {image_tensor.shape}", 45)
    return image_tensor, input_img

def save_predictions(predictions, input_img, output_dir, base_filename, isniigz):
    """
        Save predictions as NIfTI and MAT files.
        @param predictions: Model output predictions (torch.Tensor)
        @param input_img: Original input image used for predictions (nibabel Nifti1Image)
        @param output_dir: Directory to save the output files (str)
        @param base_filename: Base filename for the saved output files (str)
    """
    send_progress("Post-processing predictions", 80)
    processed_preds = torch.argmax(predictions, dim=1).detach().cpu().numpy().squeeze()
    
    # Save as .nii.gz
    send_progress("Saving NIfTI file", 85)
    pred_img = nib.Nifti1Image(processed_preds, affine=input_img.affine, header=input_img.header)
    if isniigz:
        nii_save_path = os.path.join(output_dir, f"{base_filename}_pred_DOMINO.nii.gz")
    else:
        nii_save_path = os.path.join(output_dir, f"{base_filename}_pred_DOMINO.nii")

    nib.save(pred_img, nii_save_path)
    send_progress("Files saved successfully.", 95)

def save_multiple_predictions(predictions, batch_meta, output_dir):
    """
        Save predictions as NIfTI and MAT files.
        @param predictions: Model output predictions (torch.Tensor)
        @param input_img: Original input image used for predictions (nibabel Nifti1Image)
        @param output_dir: Directory to save the output files (str)
        @param base_filename: Base filename for the saved output files (str)
    """
    for i in range(predictions.shape[0]):
        pred_np = torch.argmax(predictions[i], dim=0).cpu().numpy().squeeze()
        isniigz = os.path.basename(batch_meta["filename_or_obj"][i]).endswith(".nii.gz")
        filename = os.path.basename(batch_meta["filename_or_obj"][i]).replace(".nii", "").replace(".gz","")
        affine = batch_meta["affine"][i].numpy()
        header = nib.load(batch_meta["filename_or_obj"][i]).header

        # Save as .nii.gz
        send_progress(f"Processing outputs for input file - {i}", ".")
        if isniigz:
            nib.save(nib.Nifti1Image(pred_np, affine, header), os.path.join(output_dir, f"{filename}_pred_DOMINO.nii.gz"))
        else:
            nib.save(nib.Nifti1Image(pred_np, affine, header), os.path.join(output_dir, f"{filename}_pred_DOMINO.nii"))


def domino_predict_single_file(input_path, output_dir="output", model_path="./DOMINO.pth",
                       spatial_size=(256, 256, 256), num_classes=12, dataparallel=False, num_gpu=1,
                       a_min_value=0, a_max_value=255):
    """
        Predict segmentation for a single NIfTI image with progress updates via SSE.
        @param input_path: Path to the input NIfTI image file (str)
        @param output_dir: Directory to save the output files (str)
        @param model_path: Path to the model weights file (str)
        @param spatial_size: Size of the input images (tuple)
        @param num_classes: Number of output classes (int)
        @param dataparallel: Whether to use DataParallel (bool)
        @param num_gpu: Number of GPUs to use if dataparallel is True (int)
        @param a_min_value: Minimum intensity value for scaling (int or float)
        @param a_max_value: Maximum intensity value for scaling (int or float)
    """
    os.makedirs(output_dir, exist_ok=True)
    isniigz = os.path.basename(input_path).endswith(".nii.gz")
    base_filename = os.path.basename(input_path).split(".nii")[0]

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("cpu")
        send_progress("Using MPS backend (CPU due to ConvTranspose3d support limitations)", 5)
    else:
        send_progress(f"Using device: {device}", 5)

    # Load model
    model = load_model(model_path, spatial_size, num_classes, device)

    # Preprocess input
    image_tensor, input_img = preprocess_input(input_path, device, a_min_value, a_max_value)

    # Perform inference
    send_progress("Starting sliding window inference", 50)
    start_time = time.time()
    with torch.no_grad():
        predictions = sliding_window_inference(
            image_tensor, spatial_size, sw_batch_size=4, predictor=model, overlap=0.8
        )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    send_progress(f"Inference completed successfully in {elapsed_time:.2f} seconds.", 75)

    # Save predictions
    save_predictions(predictions, input_img, output_dir, base_filename, isniigz)
    
    send_progress("Processing completed successfully!", 99)

def domino_predict_multiple_files(input_path, output_dir="output", model_path="./DOMINO.pth",
                       spatial_size=(256, 256, 256), num_classes=12, dataparallel=False, num_gpu=1,
                       a_min_value=0, a_max_value=255):
    """
        Predict segmentation for a single NIfTI image with progress updates via SSE.
        @param input_path: Path to the input NIfTI image file (str)
        @param output_dir: Directory to save the output files (str)
        @param model_path: Path to the model weights file (str)
        @param spatial_size: Size of the input images (tuple)
        @param num_classes: Number of output classes (int)
        @param dataparallel: Whether to use DataParallel (bool)
        @param num_gpu: Number of GPUs to use if dataparallel is True (int)
        @param a_min_value: Minimum intensity value for scaling (int or float)
        @param a_max_value: Maximum intensity value for scaling (int or float)
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 1
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("cpu")
        send_progress("Using MPS backend (CPU due to ConvTranspose3d support limitations)", 5)
    else:
        send_progress(f"Using device: {device}", 5)

    datalist = load_decathlon_datalist(input_path, True, "testing")

    transforms = preprocess_datalists(a_min_value, a_max_value)

    dataset = Dataset(data=datalist, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    
    # Load model
    model = load_model(model_path, spatial_size, num_classes, device)

    # Perform inference
    send_progress("Starting sliding window inference", 50)
    for batch in dataloader:
        images = batch["image"].to(device)
        meta = batch["image"].meta
        start_time = time.time()
        with torch.no_grad():
            preds = sliding_window_inference(
                images, spatial_size, sw_batch_size=4, predictor=model, overlap=0.8
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        send_progress(f"Batch inference completed {elapsed_time:.2f} seconds",".")
        save_multiple_predictions(preds, meta, output_dir)
    
    send_progress("Processing completed successfully!", 99)

# Example usage
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Path for input file or a folder expected!")
    elif(len(sys.argv) > 3):
        print("Too many arguments...!")

    input_path = sys.argv[1]
    output_dir = "outputs"
    model_path = "DOMINO.pth"
    datalist_path = "datalist.json"

    if os.path.isdir(input_path):
        send_progress("Generating datalist", 2)
        generate_datalist(input_path)

        domino_predict_multiple_files(
            input_path=datalist_path,
            output_dir=output_dir,
            model_path=model_path,
            spatial_size=(256, 256, 256),
            num_classes=12,
            dataparallel=False,
            num_gpu=1,
            a_min_value=0,
            a_max_value=255,    
        )
    
    else:
        domino_predict_single_file(
            input_path=input_path,
            output_dir=output_dir,
            model_path=model_path,
            spatial_size=(256, 256, 256),
            num_classes=12,
            dataparallel=False,
            num_gpu=1,
            a_min_value=0,
            a_max_value=255,
        )

    send_progress("Output files generated!", 100)
