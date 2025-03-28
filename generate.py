import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from create_dataset import create_pbr_map, create_mask

import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class MultiMaskUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=2):
        super().__init__()
        
        self.base_model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,  
            activation=None  
        )
        
        # Add final activation
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.final_activation(x)


DEVICE = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 7  
NUM_CLASSES = 2  

MODEL = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
MODEL.load_state_dict(torch.load('./saved_model.pth', weights_only=True, map_location=torch.device('cpu') ))


def run_inference(image_path: str, threshold=0.5):
    
    MODEL.eval()
    
    # load data frpm path
    pbr_map = create_pbr_map(image_path, transforms.Resize((1024, 1024)))
    mask = create_mask(image_path, transforms.Resize((1024, 1024)))
    pbr_map = pbr_map / 255.0  # Normalize to [0,1]
    mask = mask / 255.0  # Normalize mask to [0,1] (assuming stored as 0-255)

    
    # Prepare input
    input_tensor = pbr_map.unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        pred_masks = MODEL(input_tensor).squeeze(0).cpu()
    
    # Denormalize input data
    input_np = pbr_map.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    input_np = input_np * 0.5 + 0.5  # Reverse normalization
    input_np = np.clip(input_np, 0, 1)  # Ensure valid range
    
    # Split channels according to your structure
    grayscale = input_np[..., 0]
    rgb1 = input_np[..., 1:4]
    rgb2 = input_np[..., 4:7]
    
    # Convert masks to numpy
    true_masks_np = mask.numpy()
    pred_masks_np = pred_masks.numpy()
    pred_masks_np = (pred_masks_np > threshold).astype(np.float32)  # Apply threshold

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, max(3, NUM_CLASSES), hspace=0.3, wspace=0.2)
    
    # Plot input channels
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(grayscale, cmap='gray')
    ax0.set_title('Grayscale Channel (0)')
    ax0.axis('off')
    
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rgb1)
    ax1.set_title('RGB Set 1 (Channels 1-3)')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(rgb2)
    ax2.set_title('RGB Set 2 (Channels 4-6)')
    ax2.axis('off')
    
    # Plot true masks
    for i in range(NUM_CLASSES):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(true_masks_np[i], cmap='gray')
        ax.set_title(f'True Mask {i}')
        ax.axis('off')
    
    # Plot predicted masks
    for i in range(NUM_CLASSES):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(pred_masks_np[i], cmap='gray')
        ax.set_title(f'Predicted Mask {i}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


run_inference("validation_data/Anzio")
