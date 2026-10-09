import argparse
import os

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from config import Config, load_config
from models import build_model


class PBRDataset(Dataset):
	def __init__(self, input_data_path, spatial_transform=None, color_transform=None):
		self.input_data_path = input_data_path
		self.spatial_transform = spatial_transform
		# Applied to the image only, after the spatial transform and the mask split, so it never
		# touches mask values: these source photos vary a lot in light/shadow/time of day (see the
		# cast-shadow bands in several training shots), and jittering brightness/contrast/saturation
		# teaches the model that degradation is a texture/color-mismatch pattern, not tied to one
		# particular lighting condition.
		self.color_transform = color_transform
		data_dir = os.path.join(self.input_data_path, "data")
		self.ids = [fname.split("data_")[1] for fname in os.listdir(data_dir) if fname.startswith("data_")]
		self.ids.sort(key=lambda x: int(x))

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, idx):
		sample_id = self.ids[idx]
		pbr_map = torch.load(os.path.join(self.input_data_path, "data", f"data_{sample_id}")).float()
		mask = torch.load(os.path.join(self.input_data_path, "masks", f"mask_{sample_id}")).float()

		pbr_map = pbr_map / 255.0  # Normalize to [0,1]

		if self.spatial_transform:
			num_pbr_channels = pbr_map.shape[0]
			stacked = torch.cat([pbr_map, mask], dim=0)
			stacked = self.spatial_transform(stacked)
			pbr_map = stacked[:num_pbr_channels]
			mask = stacked[num_pbr_channels:]

		if self.color_transform:
			pbr_map = self.color_transform(pbr_map.clamp(0, 1))

		return pbr_map, mask


def loss_fn(preds, targets):
	targets = (targets > 0.5).float()

	bce_loss = nn.BCEWithLogitsLoss(reduction="mean")(preds, targets)

	dice_loss = smp.losses.DiceLoss(
		mode="multilabel",
		smooth=100.0,  # Increased smoothness for sparse masks
		from_logits=True,  # Crucial for Sigmoid outputs!
		ignore_index=0,
	)(preds, targets)

	return dice_loss, bce_loss


def calculate_binary_metrics(preds, targets, threshold=0.5, smooth=1e-6):
	"""
	Calculate IoU, Precision, Recall, and F1 Score for binary segmentation

	Args:
	    preds: Model predictions (B, 1, H, W) - logits or probabilities
	    targets: Ground truth masks (B, 1, H, W) - binary (0s and 1s)
	    threshold: Threshold for converting predictions to binary
	    smooth: Smoothing factor to avoid division by zero

	Returns:
	    dict: Dictionary containing all metrics
	"""
	if preds.max() > 1.0:  # If logits
		preds_binary = torch.sigmoid(preds) > threshold
	else:  # If probabilities/sigmoid already applied
		preds_binary = preds > threshold

	targets_binary = targets > 0.5

	preds_flat = preds_binary.view(-1).float()
	targets_flat = targets_binary.view(-1).float()

	tp = (preds_flat * targets_flat).sum()
	fp = (preds_flat * (1 - targets_flat)).sum()
	fn = ((1 - preds_flat) * targets_flat).sum()
	tn = ((1 - preds_flat) * (1 - targets_flat)).sum()

	precision = tp / (tp + fp + smooth)
	recall = tp / (tp + fn + smooth)
	f1 = 2 * (precision * recall) / (precision + recall + smooth)

	intersection = (preds_flat * targets_flat).sum()
	union = preds_flat.sum() + targets_flat.sum() - intersection
	iou = intersection / (union + smooth)

	return {
		"iou": iou.item(),
		"precision": precision.item(),
		"recall": recall.item(),
		"f1_score": f1.item(),
		"tp": tp.item(),
		"fp": fp.item(),
		"fn": fn.item(),
		"tn": tn.item(),
	}


def train(cfg: Config):
	device = cfg.train.device
	if device == "cuda" and not torch.cuda.is_available():
		print("cfg.train.device is 'cuda' but no GPU is available, falling back to 'cpu'")
		device = "cpu"

	spatial_transform = Compose(
		[
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomVerticalFlip(p=0.5),
			# Crops a random 50-100% window and resizes it back to the target resolution: with a
			# couple dozen images, this is the cheapest way to multiply how many distinct "views"
			# the model sees of each one, and it also makes small isolated patches relatively
			# bigger in-frame, which they were under-detecting (see IMG_9306).
			transforms.RandomResizedCrop(cfg.data.resolution, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
			transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
			transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
			transforms.ElasticTransform(alpha=50.0, sigma=5.0),
		]
	)
	color_transform = Compose(
		[
			transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
			# Blacks out a small random patch of the image but leaves the mask target untouched:
			# the model has to infer that spot from context instead of leaning on one obvious
			# giveaway patch, which is the kind of over-reliance that made it miss an otherwise
			# obvious area in IMG_9306.
			transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
		]
	)
	train_dataset = PBRDataset(cfg.data.output_folder, spatial_transform=spatial_transform, color_transform=color_transform)
	train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

	# With a real held-out test set, evaluate on that instead of the training data (which only
	# tells you the model memorized what it just saw). No augmentation: metrics must reflect the
	# images as they are, not random crops/flips/color jitter of them.
	test_loader = None
	if cfg.data.test_output_folder:
		test_dataset = PBRDataset(cfg.data.test_output_folder, spatial_transform=None, color_transform=None)
		test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
		print(f"[{cfg.degrado}] Valutazione su test set reale: {len(test_dataset)} immagini ({cfg.data.test_output_folder})")
	else:
		print(f"[{cfg.degrado}] Nessun test_output_folder in config: valuto sul train set (solo per debug, non è un numero affidabile)")

	model = build_model(cfg).to(device)
	if os.path.exists(cfg.train.output_model_path):
		model.load_state_dict(torch.load(cfg.train.output_model_path, weights_only=True, map_location=device))

	# AdamW (decoupled weight decay) for a bit of regularization on a small dataset, and a cosine
	# schedule instead of a flat exponential decay: it lands the LR near zero exactly at the last
	# epoch instead of trailing off arbitrarily, which gives cleaner convergence on longer runs.
	optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.01)

	os.makedirs(os.path.dirname(cfg.train.output_model_path) or ".", exist_ok=True)
	best_path = os.path.splitext(cfg.train.output_model_path)[0] + "_best.pth"
	best_iou = -1.0

	for epoch in tqdm(range(cfg.train.epochs)):
		model.train()
		running_loss = {"loss": 0, "dice_loss": 0, "bce_loss": 0}

		for images, masks in train_loader:
			images = images.to(device)
			masks = masks.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			dice_loss, bce_loss = loss_fn(outputs, masks)
			loss = dice_loss + bce_loss
			loss.backward()
			optimizer.step()

			running_loss["loss"] += loss.item()
			running_loss["dice_loss"] += dice_loss.item()
			running_loss["bce_loss"] += bce_loss.item()

		for key in running_loss:
			running_loss[key] /= len(train_loader)

		eval_loader = test_loader if test_loader is not None else train_loader
		model.eval()
		agg = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
		with torch.no_grad():
			for images, masks in eval_loader:
				images = images.to(device)
				masks = masks.to(device)
				outputs = model(images)
				metrics = calculate_binary_metrics(outputs, masks)
				for key in agg:
					agg[key] += metrics[key]
		for key in agg:
			agg[key] /= len(eval_loader)

		scheduler.step()
		torch.save(model.state_dict(), cfg.train.output_model_path)
		if agg["iou"] > best_iou:
			best_iou = agg["iou"]
			torch.save(model.state_dict(), best_path)

		tag = "test" if test_loader is not None else "train(!)"
		print(
			f"[{cfg.degrado}] Epoch {epoch + 1}/{cfg.train.epochs} - "
			f"Loss: {running_loss['loss']:.4f}, Dice Loss: {running_loss['dice_loss']:.4f}, "
			f"BCE Loss: {running_loss['bce_loss']:.4f}"
		)
		print(
			f"[{cfg.degrado}] Epoch {epoch + 1}/{cfg.train.epochs} - "
			f"IoU({tag}): {agg['iou']:.4f}  Precision: {agg['precision']:.4f}  "
			f"Recall: {agg['recall']:.4f}  F1: {agg['f1_score']:.4f}  (best IoU so far: {best_iou:.4f})"
		)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to a per-degrado config YAML, e.g. configs/cavillature.yaml")
	args = parser.parse_args()

	train(load_config(args.config))
