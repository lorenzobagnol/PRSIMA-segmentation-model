import segmentation_models_pytorch as smp
import torch.nn as nn

from config import Config


class SegmentationModel(nn.Module):
	def __init__(self, arch: str, encoder_name: str, encoder_weights: str | None, in_channels: int):
		super().__init__()
		self.base_model = smp.create_model(
			arch=arch,
			encoder_name=encoder_name,
			encoder_weights=encoder_weights,
			in_channels=in_channels,
			classes=1,
			activation=None,
		)
		self.final_activation = nn.Sigmoid()

	def forward(self, x):
		return self.final_activation(self.base_model(x))


def build_model(cfg: Config) -> SegmentationModel:
	return SegmentationModel(
		arch=cfg.model.arch,
		encoder_name=cfg.model.encoder_name,
		encoder_weights=cfg.model.encoder_weights,
		in_channels=cfg.in_channels,
	)
