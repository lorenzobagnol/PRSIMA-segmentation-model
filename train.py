import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.io as torchio
from torchvision.io import ImageReadMode 
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
        pbr_map = torch.load(os.path.join(self.input_data_path, "data", f"data_{str(idx)}"))  # Shape: (H, W, C)
        mask = torch.load(os.path.join(self.input_data_path, "masks", f"mask_{str(idx)}")) # Grayscale

        if self.transform:
            transformed = self.transform(image=pbr_map, mask=mask)
            pbr_map = transformed['image']
            mask = transformed['mask']

        return pbr_map.float(), mask.float()
  

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


DEVICE = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 7  # Adjust based on PBR maps
NUM_CLASSES = 2  # Number of damage types
BATCH_SIZE = 4
LR = 0.0001
EPOCHS = 50

# train_transform = Compose([
#     A.RandomRotate90(),
#     A.Flip(),
#     A.Normalize(mean=[0.5]*IN_CHANNELS, std=[0.5]*IN_CHANNELS),
#     ToTensorV2(),
# ])

# Initialize dataset and dataloader (replace with your paths)
train_dataset = PBRDataset(
    input_data_path=OUTPUT_FOLDER,
    transform=None
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss, and optimizer
model = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
criterion = smp.losses.DiceLoss(mode='multiclass')
optimizer = optim.Adam(model.parameters(), lr=LR)



def loss_fn(preds, targets):
    bce_loss = nn.BCELoss()(preds, targets)
    dice_loss = smp.losses.DiceLoss(mode='binary')(preds, targets)
    return bce_loss + dice_loss

optimizer = optim.Adam(model.parameters(), lr=LR)

# Step 6: Modified Training Loop
for epoch in tqdm(range(EPOCHS)):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss/len(train_loader)}")