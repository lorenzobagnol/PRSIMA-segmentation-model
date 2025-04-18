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



def create_mask(sample_path, degradi_list, resizer):

	mask_path = os.path.join(sample_path, "Maschere")
	assert os.path.isdir(mask_path), f"Mask folder not found in {sample_path}"
	
	masks = {}
	
	for i, mask_name in enumerate(degradi_list):
		
		if os.path.isfile(os.path.join(mask_path, mask_name + ".jpg")):
			masks[mask_name] = (resizer(torchio.decode_image(os.path.join(mask_path, mask_name + ".jpg"), mode=ImageReadMode.GRAY).data) > 127).float() * 255
		else: 
			masks[mask_name] = (resizer(torch.zeros((1,)+RESOLUTION)) > 127).float() * 255

	# Stack maps to create 2-channel tensor
	mask = torch.cat([masks[mask_name] for mask_name in degradi_list], dim=0)
	print(mask.shape)
	
	return mask



if __name__=="__main__":
	resizer = torchvision.transforms.Resize(RESOLUTION)

	# Process all samples
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)
	samples = [d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))]

	degradi_list = ["Cavillature", "Macchia", "Distacco"]#, "Patina biologica", "Rigonfiamento", "Esfoliazione", "Disgregazione", "Efflorescenze"]
	degradi_dict = {deg: 0 for deg in degradi_list}
	
	for i, sample in enumerate(samples):
		sample_path = os.path.join(INPUT_FOLDER, sample)
		
		# Create and save PBR map
		pbr_map = create_pbr_map(sample_path, resizer)
		os.makedirs(os.path.join(OUTPUT_FOLDER, "data"), exist_ok=True)
		torch.save(pbr_map, os.path.join(OUTPUT_FOLDER, "data", f"data_{str(i)}"))
		
		# Create and save mask
		mask = create_mask(sample_path, degradi_list, resizer)
		for i, deg in enumerate(mask):
			if deg.sum() > 0:
				degradi_dict[degradi_list[i]] += 1
		os.makedirs(os.path.join(OUTPUT_FOLDER, "masks"), exist_ok=True)
		torch.save(mask, os.path.join(OUTPUT_FOLDER, "masks", f"mask_{str(i)}"))
		
	print(degradi_dict)

		
