import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# --- 1. PyTorch Model Definition (Generator only) ---
# This class definition must be identical to the one used during training
# to correctly load the saved state_dict.

# Hyperparameters for the GAN (must match training parameters)
LATENT_DIM = 100  # Dimension of the noise vector
NUM_CLASSES = 10  # Digits 0-9
IMG_SIZE = 28     # MNIST image size
CHANNELS = 1      # Grayscale image

# Device configuration (CPU or GPU if available)
# For inference on a web app, CPU is often sufficient and avoids GPU setup complexities.
# However, if you need speed and have a GPU available on your deployment platform, 'cuda' is an option.
DEVICE = torch.device("cpu") # Set to CPU for common web hosting environments
# If you are deploying on a platform with GPU support and have configured it:
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels

        # Embedding for the class labels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            # Input: (latent_dim + num_classes) x 1 x 1
            nn.ConvTranspose2d(latent_dim + num_classes, 256, 4, 1, 0, bias=False), # Output: 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # Output: 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # Output: 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # This must match the layer from training (padding=3)
            nn.ConvTranspose2d(64, channels, 4, 2, 3, bias=False), # Output: channels x 28 x 28
            nn.Tanh() # Output pixel values in range [-1, 1]
        )

    def forward(self, noise, labels):
        # Concatenate noise vector with label embedding
        gen_input = torch.cat((self.label_emb(labels).view(-1, self.num_classes, 1, 1),
                               noise.view(-1, self.latent_dim, 1, 1)), 1)
        return self.main(gen_input)

# --- 2. Model Loading Function ---

# Define the path where the trained model will be loaded from
MODEL_PATH = "generator_model.pth"

@st.cache_resource # This decorator caches the model to avoid reloading on every rerun
def load_generator_model():
    """
    Loads the pre-trained generator model. This function will only run once
    during the app's lifetime due to st.cache_resource.
    """
    generator = Generator(LATENT_DIM, NUM_CLASSES, IMG_SIZE, CHANNELS).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. "
                 "Please ensure it's in the same directory as `app.py`. "
                 "You need to train the model on Colab first and download 'generator_model.pth'.")
        st.stop() # Stop the app execution if model is not found

    try:
        st.write(f"Loading pre-trained generator model from {MODEL_PATH}...")
        # Load state_dict, ensuring map_location is set to DEVICE
        generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        generator.eval() # Set to evaluation mode for inference
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure the model file is valid.")
        st.stop() # Stop the app execution on loading error
    return generator

# --- 3. Streamlit Web Application UI ---
st.set_page_config(layout="centered", page_title="Handwritten Digit Generator")

st.title("Handwritten Digit Image Generator")
st.markdown("Enter a digit (0-9) and click 'Generate Images' to see 5 unique handwritten versions created by a PyTorch GAN model.")

# Load the model only once at the start of the application
generator_model = load_generator_model()

# User input for the digit
selected_digit = st.number_input(
    "Enter a digit (0-9):",
    min_value=0,
    max_value=9,
    value=0, # Default value
    step=1
)

# Button to trigger image generation
if st.button("Generate Images"):
    st.subheader(f"Generated Images for Digit {selected_digit}:")
    generated_images = []

    # Generate 5 images for the selected digit
    with torch.no_grad(): # Disable gradient calculations for inference
        for i in range(5):
            noise = torch.randn(1, LATENT_DIM, device=DEVICE) # Generate random noise
            label = torch.tensor([selected_digit], device=DEVICE) # Create tensor for the desired digit label
            generated_img_tensor = generator_model(noise, label).cpu() # Generate image and move to CPU

            # Convert tensor to PIL Image for display
            img_np = generated_img_tensor.squeeze().numpy() # Remove batch and channel dims, convert to numpy
            img_np = (img_np * 0.5 + 0.5) * 255 # Denormalize from [-1, 1] to [0, 255] and scale
            img_pil = Image.fromarray(img_np.astype(np.uint8)) # Create PIL Image
            generated_images.append(img_pil)

    # Display images in columns
    cols = st.columns(5) # Create 5 columns for images
    for i, img in enumerate(generated_images):
        with cols[i]:
            st.image(img, caption=f"Image {i+1}", use_column_width=True)

    st.success("Images generated successfully!")

st.markdown("""
---
### How to Deploy This Streamlit Application:

1.  **Train Your Model in Colab:**
    * Ensure your GAN model has been fully trained using the `mnist-gan-colab-training` Canvas in Google Colab.
    * Download the `generator_model.pth` file to your local machine from Colab using:
        ```python
        from google.colab import files
        files.download("generator_model.pth")
        ```
2.  **Create a GitHub Repository:**
    * Go to [GitHub](https://github.com/) and create a new public repository (e.g., `mnist-gan-app`).
3.  **Prepare Your Project Files:**
    * Save the entire code block above as `app.py` in your local project directory.
    * Place the `generator_model.pth` file (downloaded from Colab) in the **same directory** as `app.py`.
    * Create a `requirements.txt` file in the same directory. This file should list all Python dependencies:
        ```
        streamlit
        torch
        torchvision
        numpy
        Pillow
        ```
    * (Optional but recommended) Create a `Procfile` if you're using other platforms like Heroku, but for Streamlit Community Cloud, `app.py` is usually enough.
4.  **Push to GitHub:**
    * Initialize a Git repository in your project directory (if not already done):
        ```bash
        git init
        git add .
        git commit -m "Initial commit for MNIST GAN Streamlit app"
        git branch -M main
        git remote add origin <YOUR_GITHUB_REPO_URL>
        git push -u origin main
        ```
5.  **Deploy with Streamlit Community Cloud (Recommended):**
    * Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign up/log in.
    * Click on "New app" or "Deploy an app".
    * Connect your GitHub account and select the repository you just created.
    * Specify the branch (e.g., `main`) and the path to your Streamlit app file (e.g., `app.py`).
    * Click "Deploy!".
    * Streamlit will build your application, install dependencies, and provide a public URL for your web app.

This setup ensures your web application is independent of your Colab session and can be publicly accessible.
""")
