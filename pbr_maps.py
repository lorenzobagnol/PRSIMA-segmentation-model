import os

import torch
import torchvision.io as torchio
from torchvision.io import ImageReadMode

# Channel count and decode mode for each PBR map this project knows how to load.
# Add an entry here to make a new map usable in a config's data.pbr_maps list.
MAP_SPECS = {
	"AO": (1, ImageReadMode.GRAY),
	"Roughness": (1, ImageReadMode.GRAY),
	"Metallic": (1, ImageReadMode.GRAY),
	"Displacement": (1, ImageReadMode.GRAY),
	"Normal": (3, ImageReadMode.RGB),
	"BaseColor": (3, ImageReadMode.RGB),
}


def channels_for(pbr_maps: list[str]) -> int:
	return sum(MAP_SPECS[name][0] for name in pbr_maps)


def create_pbr_map(sample_path: str, resizer, pbr_maps: list[str]) -> torch.Tensor:
	"""Loads and stacks the requested PBR maps for one sample into a [C, H, W] tensor."""
	tensors = []
	for name in pbr_maps:
		_, mode = MAP_SPECS[name]
		image = torchio.decode_image(os.path.join(sample_path, f"{name}.jpg"), mode=mode).data
		tensors.append(resizer(image))
	return torch.cat(tensors, dim=0)


def create_mask(sample_path: str, resizer, degrado: str, resolution: tuple[int, int]) -> torch.Tensor:
	"""Loads the binary mask for a single degrado, or an all-zero mask if none is annotated."""
	mask_path = os.path.join(sample_path, "Maschere")
	assert os.path.isdir(mask_path), f"Mask folder not found in {sample_path}"

	mask_file = os.path.join(mask_path, degrado + ".jpg")
	if os.path.isfile(mask_file):
		mask = torchio.decode_image(mask_file, mode=ImageReadMode.GRAY).data
		return (resizer(mask) > 127).float()
	return torch.zeros((1,) + resolution)
