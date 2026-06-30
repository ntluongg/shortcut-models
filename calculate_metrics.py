import os
import argparse
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from tqdm import tqdm
import cv2

# Import FID utilities from the shortcut-models repo
from utils.fid import get_fid_network, fid_from_stats

def load_images_from_dir(directory, batch_size=50):
    """Generator that yields batches of images from a directory."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        images = []
        for f in batch_files:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [-1, 1] as expected by the FID network in this repo
            img = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
            images.append(img)
        
        yield np.array(images)

def compute_fid(samples_dir, stats_path, batch_size=50):
    print(f"Loading reference FID stats from {stats_path}...")
    truth_fid_stats = np.load(stats_path)
    truth_mu = truth_fid_stats['mu']
    truth_sigma = truth_fid_stats['sigma']

    print("Initializing FID network...")
    get_fid_activations = get_fid_network()
    
    activations = []
    print(f"Processing images in {samples_dir}...")
    for batch in tqdm(load_images_from_dir(samples_dir, batch_size)):
        # Ensure image is in [B, 299, 299, 3] for Inception
        # The repo uses jax.image.resize
        batch_resized = jax.image.resize(batch, (batch.shape[0], 299, 299, 3), method='bilinear', antialias=False)
        batch_resized = jnp.clip(batch_resized, -1, 1)
        
        # Get activations
        acts = get_fid_activations(batch_resized)[..., 0, 0, :]
        acts = np.array(acts)
        activations.append(acts)
        
    if len(activations) == 0:
        raise ValueError("No images found in the specified directory.")
        
    activations = np.concatenate(activations, axis=0)
    activations = activations.reshape((-1, activations.shape[-1]))
    
    print(f"Computing statistics over {activations.shape[0]} samples...")
    mu_gen = np.mean(activations, axis=0)
    sigma_gen = np.cov(activations, rowvar=False)
    
    fid_score = fid_from_stats(mu_gen, sigma_gen, truth_mu, truth_sigma)
    print(f"\n---> FID Score: {fid_score:.4f} <---")
    return fid_score

def compute_fdd(samples_dir, real_stats_dir=None):
    """
    Placeholder for Fréchet DINOv2 Distance (FDD).
    
    To implement this, you typically need to:
    1. Load torch and the dinov2_vits14 model (e.g. torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'))
    2. Extract features for all generated samples.
    3. Extract features for the REAL ImageNet validation set (since we don't have a precomputed .npz for FDD).
    4. Compute the Fréchet distance using scipy.linalg.sqrtm between the two distributions.
    
    Since computing the real ImageNet stats requires the full ImageNet validation set on disk, 
    this metric is left as a secondary step for when the real dataset is mounted.
    """
    print("\n[NOTE] FDD calculation requires processing the real ImageNet validation set first to get reference stats.")
    print("       Once you have the real ImageNet images available, you can extract DINOv2 features for both")
    print("       the real and generated sets and compute the Fréchet distance using scipy.linalg.sqrtm.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True, help="Path to directory containing generated images")
    parser.add_argument("--fid_stats", type=str, default="Shortcut Model Checkpoints/imagenet256_fidstats_ours.npz", help="Path to the reference .npz stats")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    compute_fid(args.samples_dir, args.fid_stats, args.batch_size)
    compute_fdd(args.samples_dir)
