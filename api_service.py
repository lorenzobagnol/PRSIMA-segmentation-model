import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision
import torchvision.io as torchio
from torchvision.io import ImageReadMode
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import io
import os
import tempfile
import zipfile
from typing import List
import uvicorn

class MultiMaskUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=2):
        super().__init__()
        
        self.base_model = smp.Unet(
            encoder_name="efficientnet-b4",
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

# Global variables for model
MODEL = None
DEVICE = None
IN_CHANNELS = 7  
NUM_CLASSES = 2  
RESOLUTION = (1024, 1024)

def load_model():
    """Load model once at startup"""
    global MODEL, DEVICE
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    MODEL = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
    MODEL.load_state_dict(torch.load('./saved_model.pth', weights_only=True, map_location=torch.device(DEVICE)))
    MODEL.eval()  # Set to evaluation mode
    print("Model loaded successfully!")

def create_pbr_map_from_files(ao_file, normal_file, color_file):
    """Create PBR map from uploaded files"""
    # Read files into tensors
    ao_bytes = io.BytesIO(ao_file)
    normal_bytes = io.BytesIO(normal_file)
    color_bytes = io.BytesIO(color_file)
    
    # Decode images
    ao = torchio.decode_image(ao_bytes, mode=ImageReadMode.GRAY).data
    normal = torchio.decode_image(normal_bytes, mode=ImageReadMode.RGB).data
    color = torchio.decode_image(color_bytes, mode=ImageReadMode.RGB).data
    
    # Stack maps to create 7-channel tensor
    pbr_map = torch.cat([ao, normal, color], dim=0)
    
    return pbr_map

def generate_masks(pbr_tensor, resize=True):
    """Generate masks from PBR tensor"""
    if resize:
        original_size = pbr_tensor.shape[1:]
        pbr_tensor = torchvision.transforms.Resize(RESOLUTION)(pbr_tensor)
    
    # Move to device and add batch dimension
    pbr_input = pbr_tensor.unsqueeze(0).float().to(DEVICE)
    
    # Generate prediction
    with torch.no_grad():
        output = MODEL(pbr_input)
    
    output = output.squeeze(0).cpu()
    
    # Process masks
    masks = []
    for i in range(NUM_CLASSES):
        mask = output[i, :]
        if resize:
            mask = torchvision.transforms.Resize(original_size)(mask.unsqueeze(0)).squeeze(0)
        mask = (mask > 0.5).to(torch.uint8) * 255  # Binarize the mask
        masks.append(mask)
    
    return masks

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    load_model()
    yield
    # Clean up the ML model and release the resources
    MODEL.clear()

# Initialize FastAPI app
app = FastAPI(title="PBR Mask Generation API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "PBR Mask Generation API is running", "device": DEVICE}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None, "device": DEVICE}

@app.post("/generate-masks")
async def generate_masks_endpoint(
    ao_image: UploadFile = File(..., description="AO (Ambient Occlusion) image"),
    normal_image: UploadFile = File(..., description="Normal map image"),
    basecolor_image: UploadFile = File(..., description="Base color image"),
    resize: bool = True
):
    """
    Generate segmentation masks from PBR texture maps
    """
    try:
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        for file in [ao_image, normal_image, basecolor_image]:
            if file.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")
        
        # Read file contents
        ao_content = await ao_image.read()
        normal_content = await normal_image.read()
        basecolor_content = await basecolor_image.read()
        
        # Create PBR map
        pbr_tensor = create_pbr_map_from_files(ao_content, normal_content, basecolor_content)
        
        # Generate masks
        masks = generate_masks(pbr_tensor, resize=resize)
        
        # Create temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_files = []
            
            # Save masks as PNG files
            for i, mask in enumerate(masks):
                mask_path = os.path.join(temp_dir, f"mask_{i}.png")
                torchvision.io.write_png(mask.unsqueeze(0), mask_path)
                mask_files.append(mask_path)
            
            # Create ZIP file with all masks
            zip_path = os.path.join(temp_dir, "masks.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for mask_file in mask_files:
                    zipf.write(mask_file, os.path.basename(mask_file))
            
            # Return ZIP file
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename="generated_masks.zip"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)