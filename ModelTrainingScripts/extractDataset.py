# Local Dataset Path
DATASET = "archive.zip"

# Extract Dataset with Progress Bar
import zipfile
from tqdm import tqdm  # Import tqdm for the progress bar

# Replace "/content/drive/MyDrive/deepfakeimgdetection/dataset.zip" with the local path
with zipfile.ZipFile(DATASET, 'r') as zip_ref:
    # Get the total number of files in the zip archive for progress tracking
    num_files = len(zip_ref.namelist())

    # Use tqdm to create a progress bar
    with tqdm(total=num_files, desc='Extracting', unit=' files') as pbar:
        for file in zip_ref.namelist():
            zip_ref.extract(file, "")
            pbar.update(1)  # Update progress bar