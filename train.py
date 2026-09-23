import argparse
import os

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from config import Config, load_config
from models import build_model


class PBRDataset(Dataset):
	def __init__(self, input_data_path, spatial_transform=None):
		self.input_data_path = input_data_path
		self.spatial_transform = spatial_transform
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
			transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
			transforms.ElasticTransform(alpha=50.0, sigma=5.0),
		]
	)
	train_dataset = PBRDataset(cfg.data.output_folder, spatial_transform=spatial_transform)
	train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

	model = build_model(cfg).to(device)
	if os.path.exists(cfg.train.output_model_path):
		model.load_state_dict(torch.load(cfg.train.output_model_path, weights_only=True, map_location=device))

	optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
	scheduler = ExponentialLR(optimizer, gamma=0.9)

	os.makedirs(os.path.dirname(cfg.train.output_model_path) or ".", exist_ok=True)

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

		iou = 0
		with torch.no_grad():
			for images, masks in train_loader:
				images = images.to(device)
				masks = masks.to(device)
				outputs = model(images)
				metrics = calculate_binary_metrics(outputs, masks)
				iou += metrics["iou"]
			iou /= len(train_loader)

		scheduler.step()
		torch.save(model.state_dict(), cfg.train.output_model_path)

		print(
			f"[{cfg.degrado}] Epoch {epoch + 1}/{cfg.train.epochs} - "
			f"Loss: {running_loss['loss']:.4f}, Dice Loss: {running_loss['dice_loss']:.4f}, "
			f"BCE Loss: {running_loss['bce_loss']:.4f}"
		)
		print(f"[{cfg.degrado}] Epoch {epoch + 1}/{cfg.train.epochs} - IOU: {iou:.4f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", required=True, help="Path to a per-degrado config YAML, e.g. configs/cavillature.yaml")
	args = parser.parse_args()

	train(load_config(args.config))
