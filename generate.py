import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from config import load_config
from models import build_model
from pbr_maps import MAP_SPECS, create_mask, create_pbr_map


def run_inference(cfg, sample_path: str, threshold: float = 0.5):
	device = "cpu"

	model = build_model(cfg).to(device)
	model.load_state_dict(torch.load(cfg.train.output_model_path, weights_only=True, map_location=device))
	model.eval()

	resizer = transforms.Resize(cfg.data.resolution)
	pbr_map = create_pbr_map(sample_path, resizer, cfg.data.pbr_maps)
	mask = create_mask(sample_path, resizer, cfg.degrado, cfg.data.resolution)
	pbr_map = pbr_map / 255.0

	input_tensor = pbr_map.unsqueeze(0).to(device)

	with torch.no_grad():
		pred_mask = model(input_tensor).squeeze(0).cpu()

	true_mask_np = mask.numpy()[0]
	pred_mask_np = (pred_mask.numpy()[0] > threshold).astype(np.float32)

	# Show each loaded PBR map as its own panel, whatever maps this degrado's config uses.
	channel_names, offset = [], 0
	for name in cfg.data.pbr_maps:
		n_channels, _ = MAP_SPECS[name]
		channel_names.append((name, offset, offset + n_channels))
		offset += n_channels

	input_np = pbr_map.numpy().transpose(1, 2, 0)

	fig = plt.figure(figsize=(6 * max(len(channel_names), 2), 12))
	gs = fig.add_gridspec(3, max(len(channel_names), 2), hspace=0.3, wspace=0.2)

	for i, (name, start, end) in enumerate(channel_names):
		ax = fig.add_subplot(gs[0, i])
		panel = input_np[..., start:end]
		ax.imshow(panel.squeeze(-1) if panel.shape[-1] == 1 else panel, cmap="gray" if panel.shape[-1] == 1 else None)
		ax.set_title(name)
		ax.axis("off")

	ax = fig.add_subplot(gs[1, 0])
	ax.imshow(true_mask_np, cmap="gray")
	ax.set_title(f"True mask ({cfg.degrado})")
	ax.axis("off")

	ax = fig.add_subplot(gs[2, 0])
	ax.imshow(pred_mask_np, cmap="gray")
	ax.set_title(f"Predicted mask ({cfg.degrado})")
	ax.axis("off")

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to a per-degrado config YAML, e.g. configs/cavillature.yaml")
	parser.add_argument("--sample", required=True, help="Path to a sample folder, e.g. validation_data/raw-data/Anzio")
	parser.add_argument("--threshold", type=float, default=0.5)
	args = parser.parse_args()

	run_inference(load_config(args.config), args.sample, args.threshold)
