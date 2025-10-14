import traceback
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision
import torchvision.io as torchio
from torchvision.io import ImageReadMode
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware 
import io
import os
import tempfile
from typing import List
import uvicorn
import shutil

class MultiMaskUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1): 
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

# Global variables for model
MODEL = None
DEVICE = None
IN_CHANNELS = 7  
NUM_CLASSES = 1 
RESOLUTION = (1024, 1024)

def download_model_from_gcs():
    """Download model from Google Cloud Storage if not exists locally"""
    if not os.path.exists('./saved_model.pth'):
        print("Model not found locally, downloading from GCS...")
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket('architecture-degradi-models')
            blob = bucket.blob('saved_model.pth')
            blob.download_to_filename('./saved_model.pth')
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise

def load_model():
    """Load model once at startup"""
    global MODEL, DEVICE
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    download_model_from_gcs()
    MODEL = MultiMaskUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
    MODEL.load_state_dict(torch.load('./saved_model.pth', weights_only=True, map_location=torch.device(DEVICE)))
    MODEL.eval()  
    print("Model loaded successfully!")

def normalize_tensor(tensor):
    """Normalize tensor to [0,1] range based on its dtype"""
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    elif tensor.dtype == torch.uint16:
        return tensor.float() / 65535.0
    else:
        return tensor.float()

def create_pbr_map_from_files(ao_file, normal_file, color_file):
    """Create PBR map from uploaded files"""
    # Read files into tensors
    try:
        # Convert bytes to writable tensors
        ao = torch.frombuffer(bytearray(ao_file), dtype=torch.uint8)
        normal = torch.frombuffer(bytearray(normal_file), dtype=torch.uint8)
        color = torch.frombuffer(bytearray(color_file), dtype=torch.uint8)

        # Decode images
        ao = torchio.decode_image(ao, mode=ImageReadMode.GRAY).data
        normal = torchio.decode_image(normal, mode=ImageReadMode.RGB).data
        color = torchio.decode_image(color, mode=ImageReadMode.RGB).data

        # Normalize tensors
        ao = normalize_tensor(ao)
        normal = normalize_tensor(normal)
        color = normalize_tensor(color)

        if ao.shape[1:] != normal.shape[1:] or ao.shape[1:] != color.shape[1:]:
            raise ValueError("All input images must have the same dimensions")
        
        # Stack maps to create 7-channel tensor
        pbr_map = torch.cat([ao, normal, color], dim=0)
        
        return pbr_map
    except Exception as e:
        print(f"Error in create_pbr_map_from_files: {str(e)}")
        print(traceback.format_exc())
        raise

def generate_mask(pbr_tensor, resize=True):
    """Generate single mask from PBR tensor"""
    if resize:
        original_size = pbr_tensor.shape[1:]
        pbr_tensor = torchvision.transforms.Resize(RESOLUTION)(pbr_tensor)
       
    # Move to device and add batch dimension
    pbr_input = pbr_tensor.unsqueeze(0).to(DEVICE)
    
    # Generate prediction
    with torch.no_grad():
        output = MODEL(pbr_input)
    
    output = output.squeeze(0).cpu()
    
    # Process single mask
    mask = output[0, :]  # Get the first (and only) channel
    if resize:
        mask = torchvision.transforms.Resize(original_size)(mask.unsqueeze(0)).squeeze(0)
    mask = (mask > 0.8).to(torch.uint8) * 255  # Binarize the mask
    
    return mask

def remove_temp_file(temp_file):  
    """Remove temporary file"""
    try:
        os.remove(temp_file)
        print(f"Temporary file {temp_file} removed successfully.")
    except Exception as e:
        print(f"Failed to remove temporary file {temp_file}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    load_model()
    yield
    # Clean up the ML model and release the resources
    MODEL.clear()

# Initialize FastAPI app
app = FastAPI(title="PBR Mask Generation API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware - for accessing the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "PBR Mask Generation API is running", "device": DEVICE}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None, "device": DEVICE}

@app.post("/generate-mask")  # Renamed endpoint
async def generate_mask_endpoint(
    background_tasks: BackgroundTasks,
    ao_image: UploadFile = File(..., description="AO (Ambient Occlusion) image"),
    normal_image: UploadFile = File(..., description="Normal map image"),
    basecolor_image: UploadFile = File(..., description="Base color image"),
    resize: bool = True,
):
    """
    Generate single segmentation mask from PBR texture maps
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
        
        # Generate single mask
        mask = generate_mask(pbr_tensor, resize=resize)
        
        # Create temporary file for the mask
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()  # Close the file handle so we can write to it
        
        # Save mask as PNG file
        torchvision.io.write_png(mask.unsqueeze(0), temp_file.name)
        
        # Schedule cleanup
        background_tasks.add_task(remove_temp_file, temp_file.name)
        
        # Return PNG file
        return FileResponse(
            temp_file.name,
            media_type="image/png",
            filename="generated_mask.png"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Detailed error: {str(e)}", flush=True)  # Ensure it appears in logs
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)