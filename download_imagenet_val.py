import os
from datasets import load_dataset
from tqdm import tqdm

def download_val_set():
    print("Downloading ImageNet 256 validation set (used as test set) from HuggingFace...")
    # evanarlian/imagenet_1k_resized_256 is a popular pre-resized ImageNet validation set
    dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    
    out_dir = "imagenet256_val"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving 50,000 images to {out_dir}...")
    for i, item in enumerate(tqdm(dataset, total=50000)):
        # Convert PIL image to RGB in case of grayscale, then save
        img = item['image'].convert("RGB")
        img.save(os.path.join(out_dir, f"val_{i:08d}.png"))
        
    print("Download and extraction complete!")

if __name__ == "__main__":
    download_val_set()
