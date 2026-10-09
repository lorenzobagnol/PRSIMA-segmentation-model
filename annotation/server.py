"""Dataset creation tool: upload photos and correct masks by hand, one degrado per image.

Each category (degrado or materiale) has its own engine, swappable over time without a redeploy:
- **no model yet**: SAM 3 with that category's fixed prompts (at most 3, set once when the
  category is created and never editable afterwards — changing them would silently invalidate
  every mask computed so far, so the UI doesn't offer it).
- **a trained U-Net**: used as the fast (~1-4s on CPU) default starting mask; SAM 3 stays
  available as an opt-in "aiuto extra" for images the model handles badly. Uploading a new
  checkpoint for a category (POST /api/upload-model) swaps it in immediately, no restart needed.

Images belong to exactly one category, chosen at upload time (not changeable afterwards: a photo
annotated for Distacco isn't also annotated for Crepa). A later version may re-tag an image into
several categories once there are working models per degrado to cross-check against each other.

Storage layout under --data (a local folder, or a GCS bucket mounted as a volume on Cloud Run):
    images/<id>.<ext>              original upload, untouched; <id> = first 12 hex of its SHA-256
    meta/<id>.json                 source name, test flag, category, review/compute status
    masks/<Category>/<id>.png      saved masks (an empty one = checked, degrado absent)
    categories.json                [{"name", "kind", "prompts": [...], "model": "models/<Name>.pth" | null}]
    models/<Category>.pth          trained checkpoint for a category, if any (uploaded from the UI)
    cache/<id>/work.jpg            working copy, longest side capped at WORK_MAX_SIDE
    cache/<id>/                    prob_query/semantic + prompts/ (SAM 3), prob_model.png +
                                    mask_model.png (U-Net) — all aligned to work.jpg
    uploads/                       staging for direct browser -> bucket uploads
    exports/dataset-<timestamp>.zip

meta/*.json are the source of truth; read once at startup into an in-memory index.

    python annotation/server.py --data ./annotation_store --port 7860
"""

import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

WORK_MAX_SIDE = int(os.environ.get("WORK_MAX_SIDE", 2048))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
KINDS = ["degrado", "materiale", "altro"]
MAX_PROMPTS = 3
Image.MAX_IMAGE_PIXELS = 1_000_000_000  # survey photos can be huge; PIL's default refuses ~180MP

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.environ.get("ANNOTATION_DATA", "./annotation_store"))
parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
parser.add_argument("--device", default=os.environ.get("SAM3_DEVICE"), help="cuda or cpu (default: cuda if available)")
parser.add_argument("--bucket", default=os.environ.get("BUCKET"), help="GCS bucket mounted at --data: enables direct uploads")
parser.add_argument("--no-sam", action="store_true", help="Start without SAM 3 (U-Net-equipped categories still work)")
ARGS = parser.parse_args()


def path(*parts: str) -> str:
	return os.path.join(ARGS.data, *parts)


for folder in ("images", "meta", "masks", "cache", "uploads", "exports", "models"):
	os.makedirs(path(folder), exist_ok=True)

SAM = None
if not ARGS.no_sam:
	try:
		from sam3_engine import Sam3Engine
		SAM = Sam3Engine(ARGS.device)
	except Exception as e:
		print(f"SAM 3 non disponibile ({e}): le categorie senza modello non potranno calcolare previsioni.")

app = FastAPI()
INDEX: dict[str, dict] = {}  # image id -> meta
INDEX_LOCK = threading.Lock()
UNET_CACHE: dict[str, object] = {}  # category name -> UNetEngine, lazy-loaded and hot-swappable


# --- storage helpers ------------------------------------------------------------------------

def read_json(file: str, default):
	if os.path.isfile(file):
		with open(file, encoding="utf-8") as f:
			return json.load(f)
	return default


def write_json(file: str, value) -> None:
	os.makedirs(os.path.dirname(file), exist_ok=True)
	with open(file, "w", encoding="utf-8") as f:
		json.dump(value, f, indent=1, ensure_ascii=False)


def slugify(text: str) -> str:
	slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]
	if not slug:
		raise HTTPException(400, "Testo non valido.")
	return slug


def safe_category_name(name: str) -> str:
	if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,59}", name):
		raise HTTPException(400, f"Nome non valido: {name!r} (lettere, numeri, spazi, _ -).")
	return name


def categories() -> list[dict]:
	return read_json(path("categories.json"), [])


def category(name: str) -> dict:
	for c in categories():
		if c["name"] == name:
			return c
	raise HTTPException(404, f"Categoria sconosciuta: {name}")


def unet_for(cat: dict):
	"""Lazily loads (or reloads, after an upload) the U-Net for a category; None if it has none."""
	if not cat.get("model"):
		return None
	model_path = path(cat["model"])
	cached = UNET_CACHE.get(cat["name"])
	if cached is not None and cached[0] == model_path:
		return cached[1]
	from unet_engine import UNetEngine
	engine = UNetEngine(model_path, device=ARGS.device)
	UNET_CACHE[cat["name"]] = (model_path, engine)
	return engine


def get_meta(image_id: str) -> dict:
	if image_id not in INDEX:
		raise HTTPException(404, f"Immagine sconosciuta: {image_id}")
	return INDEX[image_id]


def save_meta(meta: dict) -> None:
	write_json(path("meta", f"{meta['id']}.json"), meta)
	with INDEX_LOCK:
		INDEX[meta["id"]] = meta


def load_index() -> None:
	files = [f for f in os.listdir(path("meta")) if f.endswith(".json")]
	with ThreadPoolExecutor(32) as pool:  # parallel reads: each one is a round trip on a bucket mount
		for meta in pool.map(lambda f: read_json(path("meta", f), None), files):
			if meta:
				INDEX[meta["id"]] = meta


def work_image(image_id: str) -> Image.Image:
	"""The working copy every mask and map is aligned to; recreated from the original if missing."""
	meta, work = get_meta(image_id), path("cache", image_id, "work.jpg")
	if not os.path.isfile(work):
		original = ImageOps.exif_transpose(Image.open(path("images", image_id + meta["ext"]))).convert("RGB")
		if max(original.size) > WORK_MAX_SIDE:
			original.thumbnail((WORK_MAX_SIDE, WORK_MAX_SIDE), Image.LANCZOS)
		os.makedirs(os.path.dirname(work), exist_ok=True)
		original.save(work, quality=95)
	return Image.open(work).convert("RGB")


def to_png(prob: np.ndarray) -> Image.Image:
	return Image.fromarray((np.clip(prob, 0, 1) * 255).round().astype(np.uint8))


def mask_file(image_id: str, cat: str) -> str:
	return path("masks", cat, f"{image_id}.png")


def image_state(meta: dict) -> dict:
	image_id, cdir, cat = meta["id"], f"cache/{meta['id']}", meta["category"]
	model_computed = meta.get("model_computed", False)
	# The U-Net, when the category has one, is both faster and (once trained) more accurate than
	# raw SAM 3, so it wins as the default starting mask; SAM 3 stays one click away.
	default_mask = f"{cdir}/mask_model.png" if model_computed else f"{cdir}/mask_sam3.png"
	return {
		"name": image_id, "label": meta["source"], "test": meta["test"], "size": meta["size"],
		"category": cat, "work": f"{cdir}/work.jpg",
		"computed": meta["computed"], "model_computed": model_computed, "reviewed": meta["reviewed"],
		"dir": cdir, "mask": f"masks/{cat}/{image_id}.png" if meta["reviewed"] else default_mask,
		"model_mask": f"{cdir}/mask_model.png" if model_computed else None,
		"sam_mask": f"{cdir}/mask_sam3.png" if meta["computed"] else None,
		"prompts": [{"slug": s, **p} for s, p in meta["prompts"].items()],
	}


def require_sam():
	if SAM is None:
		raise HTTPException(400, "SAM 3 non disponibile in questo ambiente.")


def ingest(data: bytes, filename: str, cat: str) -> tuple[str, bool]:
	"""Stores one uploaded file tagged with its one category; returns (image id, already present)."""
	category(cat)
	image_id = hashlib.sha256(data).hexdigest()[:12]
	if image_id in INDEX:
		return image_id, True  # already ingested (possibly for a different category: left alone)
	pil = Image.open(io.BytesIO(data))
	pil.load()
	ext = os.path.splitext(filename)[1].lower()
	if ext not in IMAGE_EXTS:
		ext = "." + (pil.format or "jpg").lower().replace("jpeg", "jpg")
	with open(path("images", image_id + ext), "wb") as f:
		f.write(data)
	size = ImageOps.exif_transpose(pil).size
	meta = {"id": image_id, "source": filename, "ext": ext, "category": cat, "test": False, "size": list(size),
		"added": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
		"computed": False, "model_computed": False, "reviewed": False, "prompts": {}}
	save_meta(meta)
	work_image(image_id)
	return image_id, False


# --- pages and state ------------------------------------------------------------------------

@app.get("/")
def index():
	return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"), headers={"Cache-Control": "no-cache"})


THUMB_SIDE = 256


@app.get("/api/thumb/{image_id}")
def thumb(image_id: str):
	"""Small square-ish JPEG for the gallery, built once from the working copy and cached next to it
	(the working copy can be ~1MB; a gallery of 100 of them made the sidebar slow and janky)."""
	get_meta(image_id)
	file = path("cache", image_id, "thumb.jpg")
	if not os.path.isfile(file):
		img = work_image(image_id)
		img.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
		os.makedirs(os.path.dirname(file), exist_ok=True)
		img.save(file, quality=80)
	return FileResponse(file, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/state")
def state():
	images = sorted(INDEX.values(), key=lambda m: (m["added"], m["source"]))
	cats = [{**c, "has_model": bool(c.get("model"))} for c in categories()]
	return {"categories": cats, "kinds": KINDS, "max_prompts": MAX_PROMPTS, "images": [image_state(m) for m in images],
		"has_sam3": SAM is not None, "sam_device": SAM.device if SAM else "none",
		"direct_upload": bool(ARGS.bucket), "work_max_side": WORK_MAX_SIDE}


# --- categories -----------------------------------------------------------------------------

class CategoryRequest(BaseModel):
	name: str
	kind: str = "degrado"
	prompts: list[str] = []


@app.post("/api/categories")
def create_category(req: CategoryRequest):
	name = safe_category_name(req.name.strip())
	if any(c["name"] == name for c in categories()):
		raise HTTPException(400, f"Categoria «{name}» già esistente.")
	if req.kind not in KINDS:
		raise HTTPException(400, f"Tipo non valido: {req.kind}")
	prompts = [p.strip() for p in req.prompts if p.strip()][:MAX_PROMPTS]
	cats = categories()
	cats.append({"name": name, "kind": req.kind, "prompts": prompts, "model": None})
	write_json(path("categories.json"), sorted(cats, key=lambda c: c["name"].lower()))
	return {"categories": categories()}


@app.post("/api/upload-model")
async def upload_model(category_name: str = Form(...), file: UploadFile = File(...)):
	"""Uploads/replaces a category's U-Net checkpoint and swaps it in immediately (no restart)."""
	cat = category(category_name)
	rel = f"models/{cat['name']}.pth"
	data = await file.read()
	with open(path(rel), "wb") as f:
		f.write(data)
	cats = [dict(c, model=rel) if c["name"] == cat["name"] else c for c in categories()]
	write_json(path("categories.json"), cats)
	UNET_CACHE.pop(cat["name"], None)  # force a reload on next use
	try:
		unet_for(category(cat["name"]))  # load eagerly so errors surface now, not on first predict
	except Exception as e:
		raise HTTPException(400, f"Checkpoint non valido per l'architettura attesa: {e}")
	return {"categories": categories()}


# --- images ---------------------------------------------------------------------------------

def ingest_many(items, cat: str) -> dict:
	added, duplicates, errors = [], [], []
	for data, filename in items:
		try:
			image_id, existed = ingest(data, filename, cat)
			(duplicates if existed else added).append(image_id)
		except Exception as e:
			errors.append(f"{filename}: non è un'immagine leggibile ({e})")
	return {"added": added, "duplicates": duplicates, "errors": errors}


@app.post("/api/upload")
async def upload(category_name: str = Form(...), files: list[UploadFile] = File(...)):
	"""Direct upload through the server: fine locally, limited to 32MB per request on Cloud Run."""
	return ingest_many([(await f.read(), f.filename or "img") for f in files], category_name)


class UploadUrlRequest(BaseModel):
	files: list[dict]  # [{"name", "type"}]


@app.post("/api/upload-urls")
def upload_urls(req: UploadUrlRequest):
	"""Signed URLs to PUT files straight into the bucket, bypassing Cloud Run's request size limit."""
	if not ARGS.bucket:
		raise HTTPException(400, "Upload diretto disponibile solo con un bucket (--bucket).")
	import google.auth
	from google.auth.transport.requests import Request
	from google.cloud import storage

	credentials, _ = google.auth.default()
	credentials.refresh(Request())
	bucket = storage.Client(credentials=credentials).bucket(ARGS.bucket)
	out = []
	for f in req.files:
		name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(f.get("name", "img")))[-80:] or "img"
		obj = f"uploads/{uuid.uuid4().hex}-{name}"
		url = bucket.blob(obj).generate_signed_url(
			version="v4", expiration=datetime.timedelta(minutes=60), method="PUT",
			content_type=f.get("type") or "application/octet-stream",
			service_account_email=credentials.service_account_email, access_token=credentials.token,
		)
		out.append({"object": obj, "url": url})
	return {"uploads": out}


class IngestRequest(BaseModel):
	objects: list[str]
	category: str


@app.post("/api/ingest")
def ingest_uploaded(req: IngestRequest):
	"""Moves files uploaded with /api/upload-urls from uploads/ into the dataset."""
	from google.cloud import storage

	bucket = storage.Client().bucket(ARGS.bucket)
	items, blobs = [], []
	for obj in req.objects:
		if not re.fullmatch(r"uploads/[0-9a-f]{32}-[A-Za-z0-9_.-]+", obj):
			raise HTTPException(400, f"Oggetto non valido: {obj}")
		blob = bucket.blob(obj)
		# Read through the API, not the mount: the object was written behind the mount's back.
		items.append((blob.download_as_bytes(), obj.split("-", 1)[1]))
		blobs.append(blob)
	result = ingest_many(items, req.category)
	for blob in blobs:
		blob.delete()
	return result


class ImageRequest(BaseModel):
	image: str


@app.post("/api/delete-image")
def delete_image(req: ImageRequest):
	meta = get_meta(req.image)
	mf = mask_file(req.image, meta["category"])
	if os.path.isfile(mf):
		os.remove(mf)
	shutil.rmtree(path("cache", req.image), ignore_errors=True)
	os.remove(path("images", req.image + meta["ext"]))
	os.remove(path("meta", f"{req.image}.json"))
	with INDEX_LOCK:
		INDEX.pop(req.image, None)
	return {"ok": True}


class TestRequest(BaseModel):
	image: str
	test: bool


@app.post("/api/test-flag")
def set_test(req: TestRequest):
	meta = get_meta(req.image)
	meta["test"] = req.test
	save_meta(meta)
	return {"ok": True}


# --- predictions ------------------------------------------------------------------------------

def compute_prompt(meta: dict, pil: Image.Image, text: str):
	"""Runs one SAM 3 text prompt, stores its maps and records it in the meta; returns (query, semantic, detected)."""
	cdir = path("cache", meta["id"])
	query, semantic, detected = SAM.prompt_maps(meta["id"], pil, text)
	slug = slugify(text)
	os.makedirs(os.path.join(cdir, "prompts"), exist_ok=True)
	to_png(query).save(os.path.join(cdir, "prompts", f"{slug}.png"))
	to_png(semantic).save(os.path.join(cdir, "prompts", f"{slug}__semantic.png"))
	meta["prompts"][slug] = {"text": text, "semantic": True}
	return query, semantic, detected


@app.post("/api/warm")
def warm(req: ImageRequest):
	"""Called when an image is opened, so the first SAM 3 prompt or click does not wait for the
	encoder. The U-Net needs no such warm-up (~1-4s either way)."""
	require_sam()
	SAM.warm(req.image, work_image(req.image))
	return {"ok": True}


class ComputeRequest(BaseModel):
	image: str
	with_sam3: bool = False  # SAM 3 takes 25-45s on a new image vs ~1-4s for the U-Net: opt-in


@app.post("/api/compute")
def compute(req: ComputeRequest):
	"""Computes the starting mask for an image's category: its U-Net if it has one (fast,
	default), and, opt-in or if there's no U-Net at all, SAM 3 with the category's fixed prompts."""
	meta = get_meta(req.image)
	cat = category(meta["category"])
	engine = unet_for(cat)
	pil, cdir = work_image(req.image), path("cache", req.image)
	os.makedirs(cdir, exist_ok=True)
	detected = 0.0

	if engine is not None:
		prob = engine.predict(pil)
		to_png(prob).save(os.path.join(cdir, "prob_model.png"))
		mask = prob > 0.5
		to_png(mask.astype(np.float32)).save(os.path.join(cdir, "mask_model.png"))
		meta["model_computed"] = True
		detected = float(mask.mean())

	if req.with_sam3 or engine is None:
		require_sam()
		if not cat["prompts"]:
			raise HTTPException(400, f"La categoria «{cat['name']}» non ha prompt SAM 3 impostati.")
		query_total = np.zeros((pil.height, pil.width), np.float32)
		semantic_total, detected_total = np.zeros_like(query_total), np.zeros(query_total.shape, bool)
		for text in cat["prompts"]:
			query, semantic, sam_detected = compute_prompt(meta, pil, text)
			query_total, semantic_total = np.maximum(query_total, query), np.maximum(semantic_total, semantic)
			detected_total |= sam_detected
		to_png(query_total).save(os.path.join(cdir, "prob_query.png"))
		to_png(semantic_total).save(os.path.join(cdir, "prob_semantic.png"))
		to_png(detected_total.astype(np.float32)).save(os.path.join(cdir, "mask_sam3.png"))
		meta["computed"] = True
		if engine is None:
			detected = float(detected_total.mean())

	save_meta(meta)
	return {"state": image_state(meta), "detected": detected}


class SamRequest(BaseModel):
	image: str
	x: int
	y: int
	size: str = "auto"


@app.post("/api/sam")
def sam(req: SamRequest):
	require_sam()
	mask, note = SAM.point(req.image, work_image(req.image), req.x, req.y, req.size)
	ys, xs = np.nonzero(mask)
	if len(xs) == 0:
		return {"empty": True, "note": "SAM non ha trovato niente sotto il clic."}
	# Send back only the bounding box of the region, so the answer stays a few KB.
	x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
	buffer = io.BytesIO()
	Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255).save(buffer, format="PNG")
	return {"empty": False, "x": x0, "y": y0, "png": base64.b64encode(buffer.getvalue()).decode(), "note": note}


# --- saving and export ----------------------------------------------------------------------

class SaveRequest(BaseModel):
	image: str
	png: str  # base64 PNG of the full mask at working resolution, white = the image's category


@app.post("/api/save")
def save(req: SaveRequest):
	meta = get_meta(req.image)
	mask = Image.open(io.BytesIO(base64.b64decode(req.png))).convert("L")
	expected = work_image(req.image).size
	if mask.size != expected:
		raise HTTPException(400, f"Dimensione maschera {mask.size} diversa dall'immagine {expected}")
	os.makedirs(path("masks", meta["category"]), exist_ok=True)
	mask.point(lambda p: 255 if p > 127 else 0).save(mask_file(req.image, meta["category"]))
	meta["reviewed"] = True
	save_meta(meta)
	return {"state": image_state(meta)}


@app.post("/api/export")
def export():
	"""Snapshot of the saved masks in the raw-data layout create_dataset.py reads, split by the
	test flag: <split>/<id>/BaseColor.jpg + Maschere/<Category>.jpg (working resolution).
	manifest.jsonl lists each reviewed image's category. Written to exports/ in the bucket."""
	stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	name = f"exports/dataset-{stamp}.zip"
	counts = {"train": 0, "test": 0}
	with zipfile.ZipFile(path(name), "w", zipfile.ZIP_DEFLATED) as z:
		lines = []
		for meta in sorted(INDEX.values(), key=lambda m: m["id"]):
			if not meta["reviewed"]:
				continue
			split = "test" if meta["test"] else "train"
			work_image(meta["id"])  # recreates the working copy if the cache was cleared
			z.write(path("cache", meta["id"], "work.jpg"), f"{split}/{meta['id']}/BaseColor.jpg")
			jpg = io.BytesIO()
			Image.open(mask_file(meta["id"], meta["category"])).save(jpg, format="JPEG", quality=100)
			z.writestr(f"{split}/{meta['id']}/Maschere/{meta['category']}.jpg", jpg.getvalue())
			lines.append(json.dumps({"id": meta["id"], "source": meta["source"], "split": split, "category": meta["category"]}, ensure_ascii=False))
			counts[split] += 1
		z.writestr("manifest.jsonl", "\n".join(lines) + "\n")
		z.writestr("categories.json", json.dumps(categories(), indent=1, ensure_ascii=False))
	return {"url": f"/data/{name}", "note": f"Export {name}: {counts['train']} immagini di train, {counts['test']} di test."}


load_index()
app.mount("/data", StaticFiles(directory=ARGS.data), name="data")

if __name__ == "__main__":
	uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
