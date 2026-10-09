"""Lightweight wrapper around a trained per-degrado U-Net, for fast pre-labeling in the
annotation tool. Self-contained (no dependency on the training repo's config.py), so the
annotation image doesn't need the whole training pipeline, only segmentation_models_pytorch.

Inference on CPU at 1024x1024 takes about 1s, against 25-45s for SAM 3's vision encoder on a
new image, so this is meant to be the fast default; SAM 3 stays available as a fallback for
images the U-Net gets badly wrong (e.g. small isolated patches it hasn't learned yet).
"""

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class _SegmentationModel(nn.Module):
	"""Mirrors models.SegmentationModel in the training repo: same submodule names
	("base_model", "final_activation"), so a checkpoint saved there loads here unchanged."""

	def __init__(self, arch: str, encoder_name: str, in_channels: int):
		super().__init__()
		self.base_model = smp.create_model(
			arch=arch, encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=1, activation=None
		)
		self.final_activation = nn.Sigmoid()

	def forward(self, x):
		return self.final_activation(self.base_model(x))


class UNetEngine:
	def __init__(self, weights_path: str, arch: str = "unet", encoder_name: str = "resnet50",
			in_channels: int = 3, resolution: tuple[int, int] = (1024, 1024), device: str | None = None):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.resolution = resolution  # (H, W), must match what the checkpoint was trained at
		self.model = _SegmentationModel(arch, encoder_name, in_channels).to(self.device).eval()
		state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
		self.model.load_state_dict(state_dict)

	@torch.no_grad()
	def predict(self, image: Image.Image) -> np.ndarray:
		"""Returns a probability map (0-1 float) at the image's own resolution."""
		width, height = image.size
		small = image.resize(self.resolution[::-1], Image.BILINEAR)  # PIL wants (W, H)
		tensor = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
		prob = self.model(tensor)  # already sigmoid-activated
		prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
		return prob[0, 0].cpu().numpy()
