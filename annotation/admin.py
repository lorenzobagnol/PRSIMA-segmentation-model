"""Operazioni di amministrazione sull'app di annotazione (non esposte agli utenti nell'interfaccia).

    # cambia i prompt SAM 3 di una categoria e ricalcola le previsioni delle immagini NON ancora salvate
    python annotation/admin.py set-prompts "Colonizzazione biologica" "moss and lichen" "green algae stain" "plants growing on wall" --recompute

Salva prima il vecchio categories.json in --backup (il bucket non ha versioning). Le maschere già
salvate (reviewed) non vengono toccate; le immagini mai calcolate seguono il flusso normale.
"""

import argparse
import datetime
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

BUCKET = "gs://architecture-degradi-annotation"
BASE = "https://annotation-429808723098.europe-west1.run.app"
SA = "annotation-app@architecture-degradi.iam.gserviceaccount.com"
GCLOUD = os.environ.get("GCLOUD", "gcloud.cmd" if os.name == "nt" else "gcloud")
MAX_PROMPTS = 3


def gcloud(*args):
	return subprocess.run([GCLOUD, *args], capture_output=True, text=True, check=True).stdout


def iap_token() -> str:
	env = dict(l.strip().split("=", 1) for l in open(".env", encoding="utf-8") if "=" in l and not l.startswith("#"))
	cid = (env.get("IAP_CLIENT_ID") or env["AP_CLIENT_ID"]).strip().strip('"')
	return gcloud("auth", "print-identity-token", f"--impersonate-service-account={SA}", f"--audiences={cid}", "--include-email").strip()


def call(token: str, path: str, body=None, tries=12):
	for _ in range(tries):  # il primo tentativo dopo una pausa può cadere nel cold start (500)
		try:
			req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
				headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
			return json.load(urllib.request.urlopen(req, timeout=300))
		except urllib.error.HTTPError as e:
			if e.code in (400, 404):
				raise RuntimeError(f"{path}: {e.read().decode()[:200]}")
		except Exception:
			pass
		time.sleep(15)
	raise RuntimeError(f"{path}: il server non risponde")


def set_prompts(args):
	prompts = [p.strip() for p in args.prompts if p.strip()]
	if not 1 <= len(prompts) <= MAX_PROMPTS:
		raise SystemExit(f"Servono da 1 a {MAX_PROMPTS} prompt.")
	os.makedirs(args.backup, exist_ok=True)
	backup = os.path.join(args.backup, f"categories_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
	gcloud("storage", "cp", "-q", f"{BUCKET}/categories.json", backup)
	cats = json.load(open(backup, encoding="utf-8"))
	cat = next((c for c in cats if c["name"] == args.category), None)
	if cat is None:
		raise SystemExit(f"Categoria sconosciuta: {args.category!r}. Esistenti: {[c['name'] for c in cats]}")
	print(f"{args.category}: {cat['prompts']}  ->  {prompts}   (backup: {backup})")
	cat["prompts"] = prompts
	new_file = backup.replace(".json", "_new.json")
	json.dump(cats, open(new_file, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
	gcloud("storage", "cp", "-q", new_file, f"{BUCKET}/categories.json")

	if not args.recompute:
		return
	token = iap_token()
	state = call(token, "/api/state")
	todo = [i for i in state["images"] if i["category"] == args.category and i["computed"] and not i["reviewed"]]
	print(f"{len(todo)} immagini non salvate con previsione SAM 3 da ricalcolare (le salvate non si toccano).")
	for k, i in enumerate(todo, 1):
		try:
			call(token, "/api/compute", {"image": i["name"], "with_sam3": True})
			print(f"  {k}/{len(todo)} {i['label']} ok")
		except RuntimeError as e:
			print(f"  {k}/{len(todo)} {i['label']} ERRORE: {e}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="cmd", required=True)
	sp = sub.add_parser("set-prompts")
	sp.add_argument("category")
	sp.add_argument("prompts", nargs="+")
	sp.add_argument("--recompute", action="store_true")
	sp.add_argument("--backup", default="./bucket_backup")
	sp.set_defaults(fn=set_prompts)
	args = parser.parse_args()
	args.fn(args)
