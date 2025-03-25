import torch
import os
import torchvision
import torchvision.io as torchio
from torchvision.io import ImageReadMode 


# Configuration
INPUT_FOLDER = "./images-and-masks/raw-data"  # Folder with subfolders for each sample
OUTPUT_FOLDER = "./images-and-masks/torch-data"
RESOLUTION = (1024, 1024)  # Set your desired resolution

# Folder structure should be:
# INPUT_FOLDER/
#   sample_1/
#       albedo.jpg
#       normal.jpg
#       roughness.jpg
#       metallic.jpg
#       ... (other maps)
#   sample_2/
#   ...

def create_pbr_map(sample_path, resizer):
	# Load individual maps (adjust based on your texture files)
	ao = torchio.decode_image(os.path.join(sample_path, "AO.jpg"), mode=ImageReadMode.GRAY).data
	normal = torchio.decode_image(os.path.join(sample_path, "Normal.jpg"), mode=ImageReadMode.RGB).data
	color = torchio.decode_image(os.path.join(sample_path, "BaseColor.jpg"), mode=ImageReadMode.RGB).data
	
	# Resize all maps to common resolution
	ao = resizer(ao)
	normal = resizer(normal)
	color = resizer(color)
	
	# Stack maps to create 7-channel tensor
	pbr_map = torch.cat(
		[ao, normal, color],
		dim=0)
	
	return pbr_map # [Channels, RESOLUTION]

def create_mask(sample_path, resizer):
	# Load individual masks (adjust based on your texture files)
	mask_1 = torchio.decode_image(os.path.join(sample_path, "Maschere", "Cavillature.jpg"), mode=ImageReadMode.GRAY).data if os.path.isfile(os.path.join(sample_path, "Maschere", "Cavillature.jpg")) else torch.zeros((1,)+RESOLUTION) 
	mask_2 = torchio.decode_image(os.path.join(sample_path, "Maschere", "Scagliatura.jpg"), mode=ImageReadMode.GRAY).data if os.path.isfile(os.path.join(sample_path, "Maschere", "Scagliatura.jpg")) else torch.zeros((1,)+RESOLUTION) 

	# Create true masks with only 0 or 255 values
	mask_1 = (mask_1 > 127).float() * 255
	mask_2 = (mask_2 > 127).float() * 255

	# Resize all maps to common resolution
	mask_1 = resizer(mask_1)
	mask_2 = resizer(mask_2)
	
	# Stack maps to create 2-channel tensor
	mask = torch.cat(
		[mask_1, mask_2],
		dim=0)
	
	return mask # [Channels, RESOLUTION]



if __name__=="__main__":
	resizer = torchvision.transforms.Resize(RESOLUTION)

	# Process all samples
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)
	samples = [d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))]

	for i, sample in enumerate(samples):
		sample_path = os.path.join(INPUT_FOLDER, sample)
		
		# Create and save PBR map
		pbr_map = create_pbr_map(sample_path, resizer)
		os.makedirs(os.path.join(OUTPUT_FOLDER, "data"), exist_ok=True)
		torch.save(pbr_map, os.path.join(OUTPUT_FOLDER, "data", f"data_{str(i)}"))
		
		# Create and save mask
		mask = create_mask(sample_path, resizer)
		os.makedirs(os.path.join(OUTPUT_FOLDER, "masks"), exist_ok=True)
		torch.save(mask, os.path.join(OUTPUT_FOLDER, "masks", f"mask_{str(i)}"))
		
