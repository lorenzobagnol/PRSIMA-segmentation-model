import requests
import os
from pathlib import Path

def find_image_file(base_name, allowed_formats=["jpg", "jpeg", "png"]):
    """
    Find an image file with the given base name and one of the allowed formats.
    Returns the full path if found, None otherwise.
    """
    for ext in allowed_formats:
        file_path = f"{base_name}.{ext}"
        if os.path.exists(file_path):
            return file_path
    return None

def get_mime_type(file_path):
    """Get the appropriate MIME type based on file extension."""
    ext = Path(file_path).suffix.lower().lstrip('.')
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }
    return mime_types.get(ext, 'image/jpeg')

def test_api():
    # Your API endpoint
    url = "https://pbr-mask-api-429808723098.europe-west12.run.app/generate-masks"
    
    # Allowed image formats
    allowed_formats = ["jpg", "jpeg", "png"]
    
    # Base names for your image files (without extension)
    image_files = {
        'ao_image': 'AO',
        'normal_image': 'Normal', 
        'basecolor_image': 'BaseColor'
    }
    
    # Find actual file paths
    file_paths = {}
    for key, base_name in image_files.items():
        file_path = find_image_file(base_name, allowed_formats)
        if file_path:
            file_paths[key] = file_path
            print(f"✅ Found {key}: {file_path}")
        else:
            print(f"❌ Error: No image file found for {base_name} with extensions: {allowed_formats}")
            return
    
    # Prepare files for upload
    files = {}
    file_handles = []  # Keep track of file handles for cleanup
    
    try:
        for key, file_path in file_paths.items():
            file_handle = open(file_path, 'rb')
            file_handles.append(file_handle)
            
            # Get the actual filename and MIME type
            filename = os.path.basename(file_path)
            mime_type = get_mime_type(file_path)
            
            files[key] = (filename, file_handle, mime_type)
        
        # Additional parameters
        data = {
            'resize': True
        }
        
        print("\nSending request to API...")
        response = requests.post(url, files=files, data=data, timeout=300)
        
        if response.status_code == 200:
            # Save the ZIP file
            with open('generated_masks.zip', 'wb') as f:
                f.write(response.content)
            print("✅ Success! Masks saved as 'generated_masks.zip'")
            print(f"Response headers: {dict(response.headers)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The model might be taking longer to process.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    finally:
        # Close all file handles
        for file_handle in file_handles:
            file_handle.close()

def test_health():
    """Test the health endpoint first"""
    health_url = "https://pbr-mask-api-429808723098.europe-west12.run.app/health"
    
    try:
        response = requests.get(health_url, timeout=30)
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(f"Health status: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    # First check if API is healthy
    if test_health():
        print("\n" + "="*50)
        print("Testing mask generation...")
        test_api()
    else:
        print("API is not responding. Please check your deployment.")