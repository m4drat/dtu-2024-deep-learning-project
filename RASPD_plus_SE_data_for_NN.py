# Instructions for using this script on HPC:
# 1. Set the input folder path in the `image_path` variable.
#    Example: "/scratch/username/images/"
# 2. Set the base folder for saving output data in the `base_saving_path` variable.
#    Example: "/scratch/username/output/"
# 3. Ensure all necessary Python libraries are installed (numpy, Pillow, pysteps).
#    Use: `pip install numpy Pillow pysteps` or load Python modules specific to HPC.
# 4. The script saves data into three subfolders under `base_saving_path`:
#    - RASPD_raw/: For raw RAPSD data before log scaling.
#    - RASPD_log/: For RAPSD data after log scaling.
#    - SE/: For SE (squared error) data.

import os
import glob
import numpy as np
from PIL import Image
from pysteps.utils import spectral

# Function to load images
def load_images_from_folder(folder_path, size=(320, 320)):
    images = []
    image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
    image_names = []  # Store filenames
    
    if len(image_files) == 0:
        print("No images found in the folder.")
        return images, image_names

    for file_path in image_files:
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                img = img.resize(size)
                images.append(np.array(img).astype(np.float32) / 255.0)
                image_names.append(os.path.basename(file_path))  # Save only the filename
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
    
    print(f"{len(images)} images loaded.")
    return np.array(images), image_names

# Function to compute RAPSD
def calculate_rapsd(images):
    rapsd_data_raw = []  # RAPSD before log scaling
    rapsd_data_log = []  # RAPSD after log scaling
    
    for img in images:
        try:
            rapsd, frequencies = spectral.rapsd(img[..., 1], fft_method=np.fft, return_freq=True)
            valid_idx = frequencies[1:] > 0

            # Save RAPSD before log scaling
            rapsd_data_raw.append((frequencies[1:], rapsd[1:]))  

            # Log scaling
            log_freq = np.log(frequencies[1:][valid_idx])
            log_rapsd = np.log(rapsd[1:][valid_idx])
            rapsd_data_log.append((log_freq, log_rapsd))
        except Exception as e:
            print(f"Error computing RAPSD for an image: {e}")
    
    return rapsd_data_raw, rapsd_data_log

# Function to compute LS fit and SE
def perform_ls_and_se(log_freq, log_rapsd):
    coeffs = np.polyfit(log_freq, log_rapsd, 1)
    poly = np.poly1d(coeffs)
    y_pred = poly(log_freq)
    se = (log_rapsd - y_pred) ** 2
    return se

# Function to save data in specific folders
def save_data_to_specific_folder(base_path, data, folder_name, data_type, image_name):
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
    
    # Save data with the image name
    base_name = os.path.splitext(image_name)[0]  # Get filename without extension
    filename = os.path.join(folder_path, f"{base_name}_{data_type}.npy")
    np.save(filename, data)
    print(f"{data_type} data for {image_name} saved to: {filename}")

# Main program
if __name__ == "__main__":
    # Input path (change this for HPC)
    image_path = "/content/drive/MyDrive/Fake/"  # Change to your input folder
    
    # Output base path (change this for HPC)
    base_saving_path = "/content/drive/MyDrive/Fake/Data/"  # Change to your output folder

    images, image_names = load_images_from_folder(image_path)

    if len(images) > 0:
        # Compute RAPSD
        rapsd_data_raw, rapsd_data_log = calculate_rapsd(images)
        
        for idx, ((raw_freq, raw_rapsd), (log_freq, log_rapsd), image_name) in enumerate(zip(rapsd_data_raw, rapsd_data_log, image_names)):
            # Compute SE
            se = perform_ls_and_se(log_freq, log_rapsd)
            
            # Save RAPSD before log scaling
            save_data_to_specific_folder(base_saving_path, (raw_freq, raw_rapsd), "RASPD_raw", "rapsd_raw", image_name)
            
            # Save RAPSD after log scaling
            save_data_to_specific_folder(base_saving_path, (log_freq, log_rapsd), "RASPD_log", "rapsd_log", image_name)
            
            # Save SE data
            save_data_to_specific_folder(base_saving_path, se, "SE", "se", image_name)

    else:
        print("No images to process.")

# How to load saved files:
# 
# import numpy as np
# 
# # Loading RAPSD data before log scaling
# raw_freq, raw_rapsd = np.load("/content/drive/MyDrive/Fake/Data/RASPD_raw/fake_0_rapsd_raw.npy", allow_pickle=True)
# 
# # Loading RAPSD data after log scaling
# log_freq, log_rapsd = np.load("/content/drive/MyDrive/Fake/Data/RASPD_log/fake_0_rapsd_log.npy", allow_pickle=True)
# 
# # Loading SE data
# se = np.load("/content/drive/MyDrive/Fake/Data/SE/fake_0_se.npy")