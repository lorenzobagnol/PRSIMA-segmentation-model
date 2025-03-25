import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from tqdm import tqdm

OUTPUT_FOLDER = "./images-and-masks/torch-data"

class PBRDataset(Dataset):
    def __init__(self, input_data_path, pbr_channels=7, transform=None):
        self.input_data_path = input_data_path
        self.transform = transform
        self.pbr_channels = pbr_channels

    def __len__(self):
        return len([data for data in os.listdir(os.path.join(self.input_data_path, "data"))])

    def __getitem__(self, idx):
        # Load multi-channel PBR maps (albedo, normal, roughness, metallic, etc.)
        pbr_map = torch.load(os.path.join(self.input_data_path, "data", f"data_{str(idx)}")).float()  # Shape: (C, H, W)
        mask = torch.load(os.path.join(self.input_data_path, "masks", f"mask_{str(idx)}")).float() # (2, H, W)

        pbr_map = pbr_map / 255.0  # Normalize to [0,1]
        mask = mask / 255.0  # Normalize mask to [0,1] (assuming stored as 0-255)

        if self.transform:
            pbr_map = self.transform(pbr_map)

        return pbr_map, mask
  

# Step 4: Modified Model for Multi-Mask Output
class MultiMaskUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=2):
        super().__init__()
        
        self.base_model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,  # Output 2 channels
            activation=None  # We'll handle activation separately
        )
        
        # Add final activation
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.final_activation(x)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 7  # Adjust based on PBR maps
NUM_CLASSES = 2  # Number of damage types
BATCH_SIZE = 4
LR = 0.0001
EPOCHS = 50

train_transform =  Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.Normalize(mean=[0.5]*IN_CHANNELS, std=[0.5]*IN_CHANNELS),
])

# Initialize dataset and dataloader (replace with your paths)
train_dataset = PBRDataset(
    input_data_path=OUTPUT_FOLDER,
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss, and optimizer
if os.path.exists("./saved_model.pth"):
    model = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('./saved_model.pth', weights_only=True))
else:
    model = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)


def loss_fn(preds, targets):
    # Convert targets to float (if not already)
    targets = targets.float().clamp(0.0, 1.0)
    
    # Calculate class weights (per batch)
    positive_pixels = targets.sum(dim=[0, 2, 3], keepdim=True)  # (1, C, 1, 1)
    total_pixels = targets.shape[0] * targets.shape[2] * targets.shape[3]
    positive_weights = (total_pixels - positive_pixels) / (positive_pixels + 1e-6)
    
    # # Weighted BCE (handles existing Sigmoid output)
    bce_loss = nn.BCELoss(reduction='sum')(preds, targets)
    # weighted_bce = (bce_loss * positive_weights).mean()
    
    # Adjusted Dice Loss for sparse targets
    dice_loss = smp.losses.DiceLoss(
        mode='multilabel',
        smooth=100.0,  # Increased smoothness for sparse masks
        from_logits=False, # Crucial for Sigmoid outputs!
        #ignore_index=0
    )(preds, targets)
    
    return dice_loss, bce_loss

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Step 6: Modified Training Loop
for epoch in tqdm(range(EPOCHS)):
    model.train()
    running_loss = {"loss": 0, "dice_loss": 0, "bce_loss": 0}
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        dice_loss, bce_loss = loss_fn(outputs, masks)
        loss = torch.sum(dice_loss + bce_loss)
        loss.backward()
        optimizer.step()
        
        running_loss["loss"]+= loss.item()
        running_loss["dice_loss"]+= dice_loss.item()
        running_loss["bce_loss"]+= bce_loss.item()
    
    scheduler.step()
    torch.save(model.state_dict(), "./saved_model.pth")

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss['loss']:.4f}, Dice Loss: {running_loss['dice_loss']:.4f}, BCE Loss: {running_loss['bce_loss']:.4f}")