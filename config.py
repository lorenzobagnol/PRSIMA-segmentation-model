from dataclasses import dataclass, field

import yaml

from pbr_maps import channels_for


@dataclass
class DataConfig:
	input_folder: str
	output_folder: str
	pbr_maps: list[str]
	resolution: tuple[int, int] = (1024, 1024)
	use_only_notna: bool = True
	# Held-out set, never trained on: train.py evaluates on it each epoch instead of on the
	# training data itself. Built separately with create_dataset.py on a config pointing at the
	# test images (same pbr_maps/resolution). Leave unset to fall back to the old train-only eval.
	test_output_folder: str | None = None


@dataclass
class ModelConfig:
	# arch is any name smp.create_model accepts: unet, unetplusplus, manet, linknet,
	# fpn, pspnet, deeplabv3, deeplabv3plus, pan, upernet, segformer (ViT, mit_* encoders), ...
	arch: str = "unet"
	encoder_name: str = "resnet50"
	encoder_weights: str = "imagenet"


@dataclass
class TrainConfig:
	batch_size: int = 4
	lr: float = 1e-4
	weight_decay: float = 1e-4
	epochs: int = 50
	device: str = "cuda"
	output_model_path: str = "./saved_model.pth"


@dataclass
class Config:
	degrado: str
	data: DataConfig
	model: ModelConfig = field(default_factory=ModelConfig)
	train: TrainConfig = field(default_factory=TrainConfig)

	@property
	def in_channels(self) -> int:
		return channels_for(self.data.pbr_maps)


def load_config(path: str) -> Config:
	with open(path) as f:
		raw = yaml.safe_load(f)

	data_raw = dict(raw["data"])
	if "resolution" in data_raw:
		data_raw["resolution"] = tuple(data_raw["resolution"])

	return Config(
		degrado=raw["degrado"],
		data=DataConfig(**data_raw),
		model=ModelConfig(**raw.get("model", {})),
		train=TrainConfig(**raw.get("train", {})),
	)
