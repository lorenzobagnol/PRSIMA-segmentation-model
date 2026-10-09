"""Migra il bucket dal vecchio formato (meta con `categories` annidate, cache/<id>/<Cat>/...)
al nuovo (meta piatti con `category`, cache/<id>/..., models/<Cat>.pth, categories.json).

    python annotation/migrate_bucket.py --backup ./bucket_backup          # solo dry-run
    python annotation/migrate_bucket.py --backup ./bucket_backup --apply  # scrive sul bucket

Prima di sovrascrivere salva in --backup i vecchi meta/*.json e categories.json (il bucket non ha
versioning). La cache vecchia NON viene cancellata. Richiede `gcloud` autenticato.
"""

import argparse
import glob
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

BUCKET = "gs://architecture-degradi-annotation"
CATEGORY = "Distacco"
NEW_CATEGORIES = [
	{"name": "Distacco", "kind": "degrado", "prompts": ["peeling plaster", "missing plaster", "flaking paint"], "model": "models/Distacco.pth"},
	{"name": "AttiVandalici", "kind": "degrado", "prompts": ["graffiti", "spray paint", "vandalism mark"], "model": None},
]
GCLOUD = os.environ.get("GCLOUD", "gcloud.cmd" if os.name == "nt" else "gcloud")


def gcloud(*args, check=True):
	return subprocess.run([GCLOUD, "storage", *args], capture_output=True, text=True, check=check).stdout


def main(args):
	os.makedirs(os.path.join(args.backup, "meta_old"), exist_ok=True)
	os.makedirs(os.path.join(args.backup, "meta_new"), exist_ok=True)
	gcloud("cp", f"{BUCKET}/meta/*.json", os.path.join(args.backup, "meta_old") + os.sep)
	gcloud("cp", f"{BUCKET}/categories.json", os.path.join(args.backup, "categories_old.json"))

	cache = [l for l in gcloud("ls", "-r", f"{BUCKET}/cache/**").split() if f"/{CATEGORY}/" in l and l.endswith(".png")]
	has_model = {re.search(r"cache/([0-9a-f]+)/", l).group(1) for l in cache if l.endswith("/prob_model.png")}

	for f in glob.glob(os.path.join(args.backup, "meta_old", "*.json")):
		m = json.load(open(f, encoding="utf-8"))
		if "categories" not in m:
			continue  # già nel nuovo formato
		c = m["categories"][CATEGORY]
		new = {"id": m["id"], "source": m["source"], "ext": m["ext"], "category": CATEGORY, "test": m["test"], "size": m["size"],
			"added": m["added"], "computed": c.get("computed", False), "model_computed": m["id"] in has_model,
			"reviewed": c.get("reviewed", False), "prompts": c.get("prompts", {})}
		with open(os.path.join(args.backup, "meta_new", os.path.basename(f)), "w", encoding="utf-8") as out:
			json.dump(new, out, indent=1, ensure_ascii=False)
	with open(os.path.join(args.backup, "categories_new.json"), "w", encoding="utf-8") as out:
		json.dump(NEW_CATEGORIES, out, indent=1, ensure_ascii=False)
	print(f"{len(cache)} file di cache da copiare, {len(os.listdir(os.path.join(args.backup, 'meta_new')))} meta convertiti.")
	if not args.apply:
		print("Dry-run: nulla scritto sul bucket. Rilancia con --apply.")
		return

	with ThreadPoolExecutor(16) as pool:  # cache/<id>/Distacco/x -> cache/<id>/x
		list(pool.map(lambda s: gcloud("cp", "-q", s, s.replace(f"/{CATEGORY}/", "/")), cache))
	gcloud("cp", "-q", os.path.join(args.backup, "meta_new", "*.json"), f"{BUCKET}/meta/")
	gcloud("cp", "-q", f"{BUCKET}/models/distacco.pth", f"{BUCKET}/models/Distacco.pth")
	gcloud("cp", "-q", os.path.join(args.backup, "categories_new.json"), f"{BUCKET}/categories.json")
	print("Fatto.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--backup", required=True)
	parser.add_argument("--apply", action="store_true")
	main(parser.parse_args())
