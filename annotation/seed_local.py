"""Popola una cartella dati locale per provare l'interfaccia senza toccare GCP.

    python annotation/seed_local.py --data ./annotation_store_local
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil

from PIL import Image, ImageOps

WORK_MAX_SIDE = 2048


def path(root, *parts):
	return os.path.join(root, *parts)


def ingest(root, src_path, category, reviewed_mask=None, test=False):
	data = open(src_path, "rb").read()
	image_id = hashlib.sha256(data).hexdigest()[:12]
	ext = os.path.splitext(src_path)[1].lower()
	os.makedirs(path(root, "images"), exist_ok=True)
	with open(path(root, "images", image_id + ext), "wb") as f:
		f.write(data)
	pil = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
	size = pil.size
	cdir = path(root, "cache", image_id)
	os.makedirs(cdir, exist_ok=True)
	work = pil.copy()
	if max(work.size) > WORK_MAX_SIDE:
		work.thumbnail((WORK_MAX_SIDE, WORK_MAX_SIDE), Image.LANCZOS)
	work.save(path(cdir, "work.jpg"), quality=95)
	reviewed = reviewed_mask is not None
	if reviewed:
		os.makedirs(path(root, "masks", category), exist_ok=True)
		Image.open(reviewed_mask).convert("L").resize(work.size).save(path(root, "masks", category, f"{image_id}.png"))
	meta = {
		"id": image_id, "source": os.path.basename(src_path), "ext": ext, "category": category,
		"test": test, "size": list(size), "added": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
		"computed": False, "model_computed": False, "reviewed": reviewed, "prompts": {},
	}
	os.makedirs(path(root, "meta"), exist_ok=True)
	with open(path(root, "meta", f"{image_id}.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=1, ensure_ascii=False)
	return image_id


def main(args):
	root = args.data
	for folder in ("images", "meta", "masks", "cache", "uploads", "exports", "models"):
		os.makedirs(path(root, folder), exist_ok=True)

	shutil.copy(args.distacco_model, path(root, "models", "Distacco.pth"))
	categories = [
		{"name": "Distacco", "kind": "degrado", "prompts": ["peeling plaster", "missing plaster", "flaking paint"], "model": "models/Distacco.pth"},
		{"name": "AttiVandalici", "kind": "degrado", "prompts": ["graffiti", "spray paint", "vandalism mark"], "model": None},
	]
	with open(path(root, "categories.json"), "w", encoding="utf-8") as f:
		json.dump(categories, f, indent=1, ensure_ascii=False)

	added = 0
	for src in sorted(glob.glob(os.path.join(args.unlabeled, "*.jpeg"))):
		ingest(root, src, "Distacco")
		added += 1
	print(f"Distacco: {added} immagini non annotate (da predire con la U-Net nell'app)")

	print(f"\nFatto. Avvia con:\n  python annotation/server.py --data {root} --no-sam --unet-weights (ignorato, letto da categories.json)")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default="./annotation_store_local")
	parser.add_argument("--unlabeled", default=r"C:\Users\loren\Downloads\distacchi-non-annotati", help="Cartella di foto non annotate da caricare come Distacco")
	parser.add_argument("--distacco-model", default="./models_distacco_best_v2.pth", help="Checkpoint U-Net da usare per la categoria Distacco")
	main(parser.parse_args())
