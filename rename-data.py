import os
import glob


INPUT_FOLDER = "./images-and-masks/raw-data"  # Folder with subfolders for each sample

samples = [d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))]

for i, sample in enumerate(samples):
    sample_path = os.path.join(INPUT_FOLDER, sample)
        
    # Search for files ending with "AO.jpg"
    matching_files = glob.glob(os.path.join(sample_path, "*Cavity.jpg"))

    # Check if a matching file was found
    if matching_files:
        # Get the first matching file (assuming there's only one)
        old_file_path = matching_files[0]
        
        # Define the new file name
        new_file_path = os.path.join(sample_path, "Cavity.jpg")
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{old_file_path}' to '{new_file_path}'")
    else:
        print("No file ending with 'Cavity.jpg' found.")
