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

OUTPUT_FOLDER = "/content/drive/MyDrive/Colab Notebooks/torch-data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 7  # Adjust based on PBR maps
NUM_CLASSES = 1  # Number of damage types
BATCH_SIZE = 4
LR = 0.0001
EPOCHS = 50



class PBRDataset(Dataset):
    def __init__(self, input_data_path, pbr_channels=7, spatial_transform=None, color_transform=None):
        self.input_data_path = input_data_path
        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        self.pbr_channels = pbr_channels
        # Build explicit list of available IDs
        data_dir = os.path.join(self.input_data_path, "data")
        self.ids = [fname.split("data_")[1] for fname in os.listdir(data_dir) if fname.startswith("data_")]
        self.ids = [os.path.splitext(x)[0] for x in self.ids]  # remove file extension if any
        self.ids.sort(key=lambda x: int(x))  # ensure correct order numerically

    def __len__(self):
        return len([data for data in os.listdir(os.path.join(self.input_data_path, "data"))])

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        # Load multi-channel PBR maps (albedo, normal, roughness, metallic, etc.)
        pbr_map = torch.load(os.path.join(self.input_data_path, "data", f"data_{str(sample_id)}")).float()  # Shape: (C, H, W)
        mask = torch.load(os.path.join(self.input_data_path, "masks", f"mask_{str(sample_id)}")).float() # (NUM_CLASSES, H, W)

        pbr_map = pbr_map / 255.0  # Normalize to [0,1]

        if self.spatial_transform:
            # Apply spatial transformations
            stacked = torch.cat([pbr_map, mask], dim=0)
            stacked = self.spatial_transform(stacked)
            # Split back into separate tensors
            pbr_map = stacked[:self.pbr_channels]
            mask = stacked[self.pbr_channels:]

        if self.color_transform:
            pbr_map = self.color_transform(pbr_map)

        return pbr_map, mask
  

# Model for Multi-Mask Output
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


# Initialize dataset and dataloader 
spatial_transform =  Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.ElasticTransform(alpha=50.0, sigma=5.0),
])
color_transform = Compose([
    transforms.Normalize(mean=[0.5]*IN_CHANNELS, std=[0.5]*IN_CHANNELS),
])
train_dataset = PBRDataset(
    input_data_path=OUTPUT_FOLDER,
    pbr_channels=IN_CHANNELS,
    spatial_transform=spatial_transform,
    color_transform=None,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
if os.path.exists("./saved_model.pth"):
    model = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('./saved_model.pth', weights_only=True))
else:
    model = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)

# Loss function
def loss_fn(preds, targets):
    # Convert targets to float (if not already)
    targets = (targets>0.5).float()
    
    # Calculate class weights (per batch)
    # positive_pixels = targets.sum(dim=[0, 2, 3], keepdim=True)  # (1, C, 1, 1)
    # total_pixels = targets.shape[0] * targets.shape[2] * targets.shape[3]
    # positive_weights = (total_pixels - positive_pixels) / (positive_pixels + 1e-6)
    
    # Weighted BCE (handles existing Sigmoid output)
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')(preds, targets)
    # weighted_bce = (bce_loss * positive_weights).mean()
    
    # Adjusted Dice Loss for sparse targets
    dice_loss = smp.losses.DiceLoss(
        mode='multilabel',
        smooth=100.0,  # Increased smoothness for sparse masks
        from_logits=True, # Crucial for Sigmoid outputs!
        ignore_index=0
    )(preds, targets)
    
    return dice_loss, bce_loss


# Metric Calculation
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
    # Convert predictions to binary
    if preds.max() > 1.0:  # If logits
        preds_binary = torch.sigmoid(preds) > threshold
    else:  # If probabilities/sigmoid already applied
        preds_binary = preds > threshold
    
    # Ensure targets are binary
    targets_binary = targets > 0.5
    
    # Flatten tensors for easier computation
    preds_flat = preds_binary.view(-1).float()
    targets_flat = targets_binary.view(-1).float()
    
    # Calculate True Positives, False Positives, False Negatives, True Negatives
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum()
    
    # Calculate metrics
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    # IoU (Intersection over Union)
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    iou = intersection / (union + smooth)
    
    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item(),
        'tn': tn.item()
    }


# Start Training
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in tqdm(range(EPOCHS)):
    model.train()
    running_loss = {"loss": 0, "dice_loss": 0, "bce_loss": 0}
    
    # train
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        dice_loss, bce_loss = loss_fn(outputs, masks)
        loss = dice_loss + bce_loss
        loss.backward()
        optimizer.step()
        
        running_loss["loss"]+= loss.item()
        running_loss["dice_loss"]+= dice_loss.item()
        running_loss["bce_loss"]+= bce_loss.item()
    
    for key in running_loss:
        running_loss[key] /= len(train_loader)

    # evaluate

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        iou = 0
        with torch.no_grad():
            outputs = model(images)
            metrics = calculate_binary_metrics(outputs, masks)
            iou += metrics['iou']
        iou /= len(train_loader)
    
    
    scheduler.step()
    torch.save(model.state_dict(), "./saved_model.pth")

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss['loss']:.4f}, Dice Loss: {running_loss['dice_loss']:.4f}, BCE Loss: {running_loss['bce_loss']:.4f}")
    print(f"Epoch {epoch+1}/{EPOCHS} - IOU: {iou:.4f}")