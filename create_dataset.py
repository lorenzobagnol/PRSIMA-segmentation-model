import torch
import os
import torchvision
import torchvision.io as torchio
from torchvision.io import ImageReadMode 
import shutil


# Configuration
INPUT_FOLDER = "./images-and-masks/raw-data"  # Folder with subfolders for each sample
OUTPUT_FOLDER = "./images-and-masks/torch-data"
RESOLUTION = (1024, 1024)  # Set your desired resolution
DEGRADI_LIST = ["Cavillature"]#, "Distacco",  "Macchia", "Patina biologica", "Rigonfiamento", "Esfoliazione", "Disgregazione", "Efflorescenze"]
PBR_MAPS = ["AO", "Normal", "BaseColor"]  # TODO change also in the code below
USE_ONLY_NOTNA = True

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

	mask_path = os.path.join(sample_path, "Maschere")
	assert os.path.isdir(mask_path), f"Mask folder not found in {sample_path}"
	
	masks = {}
	
	for i, mask_name in enumerate(DEGRADI_LIST):
		
		if os.path.isfile(os.path.join(mask_path, mask_name + ".jpg")):
			masks[mask_name] = (resizer(torchio.decode_image(os.path.join(mask_path, mask_name + ".jpg"), mode=ImageReadMode.GRAY).data) > 127).float() 
		else: 
			masks[mask_name] = (resizer(torch.zeros((1,)+RESOLUTION)) > 127).float() 

	if len(DEGRADI_LIST) == 1:
		return masks[DEGRADI_LIST[0]]

	# Cat maps in a single 2-channel tensor
	mask = torch.cat([masks[mask_name] for mask_name in DEGRADI_LIST], dim=0)

	return mask



if __name__=="__main__":
	resizer = torchvision.transforms.Resize(RESOLUTION)

	# Process all samples
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)
	# delete folder if already exists
	if os.path.exists(os.path.join(OUTPUT_FOLDER, "data")):
		shutil.rmtree(os.path.join(OUTPUT_FOLDER, "data"))
	if os.path.exists(os.path.join(OUTPUT_FOLDER, "masks")):
		shutil.rmtree(os.path.join(OUTPUT_FOLDER, "masks"))
	os.makedirs(os.path.join(OUTPUT_FOLDER, "data"), exist_ok=True)
	os.makedirs(os.path.join(OUTPUT_FOLDER, "masks"), exist_ok=True)

	degradi_dict = {deg: 0 for deg in DEGRADI_LIST}
	
	samples = [d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))]
	for i, sample in enumerate(samples):
		sample_path = os.path.join(INPUT_FOLDER, sample)
		
		# Create PBR map
		pbr_map = create_pbr_map(sample_path, resizer)
		
		# Create mask
		mask = create_mask(sample_path, resizer)

		for deg_index, deg in enumerate(mask):
			if deg.sum() > 0:
				degradi_dict[DEGRADI_LIST[deg_index]] += 1
		
		# Check if there is a degradation in the mask
		if len(DEGRADI_LIST)==1 and USE_ONLY_NOTNA and torch.sum(mask) == 0:
			continue

		# Save tensors
		torch.save(pbr_map, os.path.join(OUTPUT_FOLDER, "data", f"data_{str(i)}"))
		torch.save(mask, os.path.join(OUTPUT_FOLDER, "masks", f"mask_{str(i)}"))
		
	print(degradi_dict)

		
