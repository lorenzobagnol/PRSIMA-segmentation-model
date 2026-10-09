"""Visual comparison of predicted vs. true masks on a held-out set: saves one overlay PNG per
sample (true mask green, prediction red, overlap yellow) plus a contact sheet with all of them.

    python compare_predictions.py --config configs/distacco.yaml --sample-root images-and-masks/annotated/test
"""

import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from config import load_config
from models import build_model
from pbr_maps import create_mask, create_pbr_map


def overlay(base: Image.Image, true_mask: np.ndarray, pred_mask: np.ndarray) -> Image.Image:
	"""true-only -> green (missed), pred-only -> red (false positive), both -> yellow (correct)."""
	rgb = np.asarray(base.convert("RGB"), dtype=np.float32)
	color = np.zeros_like(rgb)
	color[..., 0] = pred_mask * 255  # red channel: prediction
	color[..., 1] = true_mask * 255  # green channel: ground truth
	mask_any = (true_mask | pred_mask).astype(np.float32)[..., None]
	blended = rgb * (1 - 0.55 * mask_any) + color * (0.55 * mask_any)
	return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def main(args):
	cfg = load_config(args.config)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = build_model(cfg).to(device)
	model.load_state_dict(torch.load(args.weights or cfg.train.output_model_path, weights_only=True, map_location=device))
	model.eval()

	resizer = transforms.Resize(cfg.data.resolution)
	samples = sorted(d for d in os.listdir(args.sample_root) if os.path.isdir(os.path.join(args.sample_root, d)))
	os.makedirs(args.output_folder, exist_ok=True)

	rows = []
	for sample in samples:
		sample_path = os.path.join(args.sample_root, sample)
		pbr_map = create_pbr_map(sample_path, resizer, cfg.data.pbr_maps)
		true_mask = create_mask(sample_path, resizer, cfg.degrado, cfg.data.resolution).numpy()[0] > 0.5

		input_tensor = (pbr_map / 255.0).unsqueeze(0).to(device)
		with torch.no_grad():
			pred = model(input_tensor).squeeze(0).cpu().numpy()[0]
		pred_mask = pred > args.threshold

		# BaseColor is always present for this dataset; show it as the base photo.
		base = Image.open(os.path.join(sample_path, "BaseColor.jpg")).convert("RGB").resize(cfg.data.resolution[::-1])
		img = overlay(base, true_mask, pred_mask)

		intersection = (true_mask & pred_mask).sum()
		union = (true_mask | pred_mask).sum()
		iou = intersection / union if union else 1.0
		out_path = os.path.join(args.output_folder, f"{sample}.jpg")
		img.save(out_path, quality=90)
		rows.append((sample, iou, out_path))
		print(f"{sample}: IoU {iou:.3f}  (verde=solo vero, rosso=solo predetto, giallo=corretto)")

	# Contact sheet, sorted worst-to-best so the problems surface first.
	rows.sort(key=lambda r: r[1])
	cols = 3
	cell = 340
	sheet = Image.new("RGB", (cols * cell, ((len(rows) + cols - 1) // cols) * (cell + 24)), (17, 17, 17))
	from PIL import ImageDraw
	draw = ImageDraw.Draw(sheet)
	for i, (sample, iou, path) in enumerate(rows):
		thumb = Image.open(path).resize((cell, cell))
		x, y = (i % cols) * cell, (i // cols) * (cell + 24)
		sheet.paste(thumb, (x, y))
		draw.text((x + 4, y + cell + 4), f"{sample}  IoU={iou:.3f}", fill=(255, 255, 255))
	sheet.save(os.path.join(args.output_folder, "_contact_sheet.jpg"), quality=85)
	print(f"\nContact sheet: {os.path.join(args.output_folder, '_contact_sheet.jpg')}")
	print(f"IoU medio: {np.mean([r[1] for r in rows]):.3f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True)
	parser.add_argument("--sample-root", required=True, help="Folder of sample subfolders, e.g. images-and-masks/annotated/test")
	parser.add_argument("--weights", default=None, help="Override cfg.train.output_model_path (e.g. the _best.pth checkpoint)")
	parser.add_argument("--output-folder", default="./predictions_review")
	parser.add_argument("--threshold", type=float, default=0.5)
	main(parser.parse_args())
