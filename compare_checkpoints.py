"""Visual A/B between two checkpoints on the same (unlabeled) photos: no ground truth here, just
old vs new prediction side by side, plus an overlay (blu=solo vecchio, rosso=solo nuovo,
viola=entrambi concordano).

    python compare_checkpoints.py --config configs/distacco.yaml --sample-root <folder con le foto>
        --old-weights models_v2_76test8/distacco_best.pth --new-weights models/distacco.pth
"""

import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from config import load_config
from models import build_model


def predict(model, device, small: Image.Image):
	pbr_map = torch.from_numpy(np.asarray(small, dtype=np.float32)).permute(2, 0, 1) / 255.0
	with torch.no_grad():
		pred = model(pbr_map.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()[0]
	return pred > 0.5


def overlay(base: Image.Image, old_mask: np.ndarray, new_mask: np.ndarray) -> Image.Image:
	rgb = np.asarray(base.convert("RGB"), dtype=np.float32)
	color = np.zeros_like(rgb)
	color[..., 0] = new_mask * 255  # red: new only (or part of agreement -> magenta with blue)
	color[..., 2] = old_mask * 255  # blue: old only
	mask_any = (old_mask | new_mask).astype(np.float32)[..., None]
	blended = rgb * (1 - 0.55 * mask_any) + color * (0.55 * mask_any)
	return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def side_by_side(base: Image.Image, old_mask: np.ndarray, new_mask: np.ndarray, both: Image.Image) -> Image.Image:
	def tint(mask, rgb):
		im = np.asarray(base.convert("RGB"), dtype=np.float32)
		color = np.zeros_like(im); color[..., 0], color[..., 1], color[..., 2] = rgb
		m = mask.astype(np.float32)[..., None]
		return Image.fromarray((im * (1 - 0.55 * m) + color * (0.55 * m)).clip(0, 255).astype(np.uint8))
	old_img, new_img = tint(old_mask, (0, 110, 255)), tint(new_mask, (255, 60, 0))
	w, h = base.size
	sheet = Image.new("RGB", (w * 3 + 20, h + 36), (17, 17, 17))
	for i, (im, label) in enumerate([(old_img, "vecchio (0.8366, 67+8 img)"), (new_img, "nuovo (76 img)"), (both, "sovrapposto: blu=solo vecchio rosso=solo nuovo viola=entrambi")]):
		sheet.paste(im, (i * (w + 10), 0))
	from PIL import ImageDraw
	draw = ImageDraw.Draw(sheet)
	for i, label in enumerate(["vecchio (0.8366)", "nuovo (76 img)", "blu=solo vecchio rosso=solo nuovo viola=entrambi"]):
		draw.text((i * (w + 10) + 4, h + 8), label, fill=(255, 255, 255))
	return sheet


def main(args):
	cfg = load_config(args.config)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	resizer = transforms.Resize(cfg.data.resolution)

	old_model = build_model(cfg).to(device).eval()
	old_model.load_state_dict(torch.load(args.old_weights, map_location=device, weights_only=True))
	new_model = build_model(cfg).to(device).eval()
	new_model.load_state_dict(torch.load(args.new_weights, map_location=device, weights_only=True))

	os.makedirs(args.output_folder, exist_ok=True)
	files = sorted(f for f in os.listdir(args.sample_root) if f.lower().endswith((".jpg", ".jpeg", ".png")))
	for fname in files:
		path = os.path.join(args.sample_root, fname)
		base = resizer(Image.open(path).convert("RGB"))
		old_mask = predict(old_model, device, base)
		new_mask = predict(new_model, device, base)
		both = overlay(base, old_mask, new_mask)
		sheet = side_by_side(base, old_mask, new_mask, both)
		name = os.path.splitext(fname)[0]
		sheet.save(os.path.join(args.output_folder, f"{name}.jpg"), quality=88)
		print(f"{fname}: vecchio {old_mask.mean():.1%}, nuovo {new_mask.mean():.1%}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True)
	parser.add_argument("--sample-root", required=True)
	parser.add_argument("--old-weights", required=True)
	parser.add_argument("--new-weights", required=True)
	parser.add_argument("--output-folder", default="./checkpoint_compare")
	main(parser.parse_args())
