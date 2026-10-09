"""SAM 3 wrapper for the annotation tool, usable on CPU.

The expensive part of SAM 3 is the vision encoder (~23s per image on 4 CPU cores, well under a
second on GPU). It is computed once per image and reused: text prompts on a cached image take
~2.4s on CPU, point prompts ~0.1s. Only the most recent image is cached, which is enough for a
single user and keeps memory flat.
"""

import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

SAM_SIZES = ["auto", "piccola", "media", "grande"]


class Sam3Engine:
	def __init__(self, device: str | None = None):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		if self.device == "cpu":
			# os.cpu_count() reports the host's cores inside containers; use the ones we may run on.
			threads = int(os.environ.get("SAM3_THREADS", 0))
			if not threads:
				threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
			torch.set_num_threads(threads)
		self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device).eval()
		self.processor = Sam3Processor.from_pretrained("facebook/sam3")
		self.tracker = Sam3TrackerModel.from_pretrained("facebook/sam3").to(self.device).eval()
		self.tracker_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
		# One request at a time: two concurrent forwards would only slow each other down on CPU.
		self.lock = threading.Lock()
		self._vision = (None, None)   # (cache key, vision features for text prompts)
		self._tracker = (None, None)  # (cache key, image embeddings for point prompts)

	def _vision_features(self, key: str, image: Image.Image):
		if self._vision[0] != key:
			pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
			self._vision = (key, self.model.get_vision_features(pixel_values=pixel_values))
		return self._vision[1]

	def _tracker_embeddings(self, key: str, pixel_values: torch.Tensor):
		if self._tracker[0] != key:
			self._tracker = (key, self.tracker.get_image_embeddings(pixel_values))
		return self._tracker[1]

	@torch.no_grad()
	def warm(self, key: str, image: Image.Image) -> None:
		"""Precomputes both caches for an image, so the first prompt or click is fast."""
		with self.lock:
			self._vision_features(key, image)
			inputs = self.tracker_processor(images=image, return_tensors="pt").to(self.device)
			self._tracker_embeddings(key, inputs["pixel_values"])

	@torch.no_grad()
	def prompt_maps(self, key: str, image: Image.Image, prompt: str, threshold: float = 0.4,
			max_instance_area: float = 0.5, min_query_score: float = 0.02):
		"""Returns (query_prob, semantic_prob, detected) as HxW numpy arrays for one text prompt.

		query_prob is max over queries of sigmoid(mask_logit) * instance_score, so it also keeps
		what SAM 3 saw below the detection threshold; detected is the usual thresholded result.
		"""
		height, width = image.height, image.width
		with self.lock:
			vision = self._vision_features(key, image)
			text = self.processor(text=prompt, return_tensors="pt").to(self.device)
			outputs = self.model(vision_embeds=vision, input_ids=text["input_ids"], attention_mask=text["attention_mask"])

		scores = outputs.pred_logits[0].sigmoid() * outputs.presence_logits[0].sigmoid()
		query_prob = torch.zeros((height, width), device=self.device)
		detected = torch.zeros((height, width), dtype=torch.bool, device=self.device)
		# Most of the 200 queries have ~0 score: upsampling only the rest keeps memory small.
		candidates = (scores > min_query_score).nonzero().flatten()
		if len(candidates):
			masks = F.interpolate(
				outputs.pred_masks[0, candidates].sigmoid().unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
			)[0]
			# An instance covering most of the photo is the whole wall, not the thing we look for.
			keep = (masks > 0.5).float().mean(dim=(1, 2)) <= max_instance_area
			masks, kept = masks[keep], scores[candidates][keep]
			if len(masks):
				query_prob = (masks * kept[:, None, None]).amax(dim=0)
				confident = masks[kept >= threshold]
				if len(confident):
					detected = (confident > 0.5).any(dim=0)

		semantic_prob = F.interpolate(outputs.semantic_seg.sigmoid(), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
		return query_prob.cpu().numpy(), semantic_prob.cpu().numpy(), detected.cpu().numpy()

	@torch.no_grad()
	def point(self, key: str, image: Image.Image, x: int, y: int, size: str = "auto"):
		"""SAM 3 point prompt: returns (mask HxW bool, note)."""
		with self.lock:
			inputs = self.tracker_processor(images=image, input_points=[[[[x, y]]]], input_labels=[[[1]]], return_tensors="pt").to(self.device)
			embeddings = self._tracker_embeddings(key, inputs["pixel_values"])
			outputs = self.tracker(
				input_points=inputs["input_points"], input_labels=inputs["input_labels"],
				image_embeddings=embeddings, multimask_output=True,
			)
		masks = self.tracker_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0].numpy() > 0.5
		ious = outputs.iou_scores[0, 0].cpu().numpy()
		areas = masks.reshape(len(masks), -1).mean(axis=1)
		if size == "auto":
			# Highest predicted IoU, ignoring whole-wall answers when a smaller one exists.
			allowed = [i for i in range(len(masks)) if areas[i] <= 0.5] or list(range(len(masks)))
			index = max(allowed, key=lambda i: ious[i])
		else:
			index = int(np.argsort(areas)[SAM_SIZES.index(size) - 1])
		return masks[index], f"SAM: oggetto {areas[index]:.1%} dell'immagine (IoU stimato {ious[index]:.2f})."
