import requests

# URL of the pre-trained model
model_url = "https://github.com/hairymax/Face-AntiSpoofing/releases/download/v1.0/AntiSpoofing_bin_128.pt"

# Local file name to save the downloaded model
model_path = "../models"

print("Downloading model...")
response = requests.get(model_url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # Write chunks to the file
                f.write(chunk)
    print(f"Model downloaded and saved as '{model_path}'")
else:
    print(f"Failed to download model. Status code: {response.status_code}")
