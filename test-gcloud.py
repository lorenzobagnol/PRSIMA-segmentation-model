import requests
import os

def test_api():
    # Your API endpoint
    url = "https://pbr-mask-api-429808723098.europe-west12.run.app/generate-masks"
    
    # Path to your image files
    ao_path = "AO.jpg"  # Replace with your actual file path
    normal_path = "Normal.jpg"  # Replace with your actual file path
    basecolor_path = "BaseColor.jpg"  # Replace with your actual file path
    
    # Check if files exist
    for file_path in [ao_path, normal_path, basecolor_path]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found!")
            return
    
    # Prepare files for upload
    files = {
        'ao_image': ('AO.jpg', open(ao_path, 'rb'), 'image/jpeg'),
        'normal_image': ('Normal.jpg', open(normal_path, 'rb'), 'image/jpeg'),
        'basecolor_image': ('BaseColor.jpg', open(basecolor_path, 'rb'), 'image/jpeg')
    }
    
    # Additional parameters
    data = {
        'resize': True
    }
    
    try:
        print("Sending request to API...")
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
        # Close file handles
        for file_tuple in files.values():
            file_tuple[1].close()

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