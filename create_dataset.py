import argparse
import os
import shutil

import torch
import torchvision

from config import load_config
from pbr_maps import create_mask, create_pbr_map

# Folder structure should be:
# INPUT_FOLDER/
#   sample_1/
#       AO.jpg
#       Normal.jpg
#       BaseColor.jpg
#       ...
#       Maschere/
#           <Degrado>.jpg
#   sample_2/
#   ...


def build_dataset(cfg):
	resizer = torchvision.transforms.Resize(cfg.data.resolution)

	os.makedirs(cfg.data.output_folder, exist_ok=True)
	data_dir = os.path.join(cfg.data.output_folder, "data")
	masks_dir = os.path.join(cfg.data.output_folder, "masks")
	if os.path.exists(data_dir):
		shutil.rmtree(data_dir)
	if os.path.exists(masks_dir):
		shutil.rmtree(masks_dir)
	os.makedirs(data_dir, exist_ok=True)
	os.makedirs(masks_dir, exist_ok=True)

	num_with_degrado = 0
	samples = [d for d in os.listdir(cfg.data.input_folder) if os.path.isdir(os.path.join(cfg.data.input_folder, d))]
	for i, sample in enumerate(samples):
		sample_path = os.path.join(cfg.data.input_folder, sample)

		pbr_map = create_pbr_map(sample_path, resizer, cfg.data.pbr_maps)
		mask = create_mask(sample_path, resizer, cfg.degrado, cfg.data.resolution)

		has_degrado = mask.sum() > 0
		if has_degrado:
			num_with_degrado += 1

		if cfg.data.use_only_notna and not has_degrado:
			continue

		torch.save(pbr_map, os.path.join(data_dir, f"data_{i}"))
		torch.save(mask, os.path.join(masks_dir, f"mask_{i}"))

	print(f"{cfg.degrado}: {num_with_degrado}/{len(samples)} samples with an annotated mask")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to a per-degrado config YAML, e.g. configs/cavillature.yaml")
	args = parser.parse_args()

	build_dataset(load_config(args.config))
