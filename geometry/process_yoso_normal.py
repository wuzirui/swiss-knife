"""
Normal Map Generator using StableNormal

This script processes a directory of images to generate normal maps using the StableNormal model.
It supports three different processing modes:
- object: Masks out background using alpha channel if available, otherwise uses birefnet
- outdoor: Uses Mask2Former to mask out sky and plants
- indoor: Processes the entire image without masking

Requirements:
    pip install torch torchvision
    pip install pillow tqdm
    pip install diffusers transformers einops
    
Usage:
    python process_yoso_normal.py --input_image_dir path/to/input/images \
                                 --output_image_dir path/to/output/normals \
                                 --data_type outdoor

Example:
    python process_yoso_normal.py --input_image_dir ./photos \
                                 --output_image_dir ./normal_maps \
                                 --data_type object

The script will:
1. Recursively find all .png and .jpg images in the input directory
2. Generate normal maps using the specified processing mode
3. Save results in the output directory, maintaining the input directory structure
4. Display a progress bar and handle any processing errors
"""

import argparse
from pathlib import Path
import torch
from PIL import Image
import tqdm

parser = argparse.ArgumentParser(description="Generate normal maps from images using StableNormal")
parser.add_argument("--input_image_dir", type=str, required=True,
                    help="Directory containing input images (will search recursively)")
parser.add_argument("--output_image_dir", type=str, required=True,
                    help="Directory where normal maps will be saved")
parser.add_argument("--data_type", type=str, default="outdoor", 
                    choices=["object", "outdoor", "indoor"],
                    help="Type of data to process: object (masks background), outdoor (masks sky/plants), indoor (no masking)")

args = parser.parse_args()
print(args)
input_image_dir = Path(args.input_image_dir)
output_dir = Path(args.output_image_dir)
output_dir.mkdir(exist_ok=True, parents=True)
image_filenames = sorted(list(input_image_dir.glob("**/*.png")))
image_filenames += sorted(list(input_image_dir.glob("**/*.jpg")))
print(f"Found {len(image_filenames)} images to process.")

# Create predictor instance
print("Loading StableNormal model...")
predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", 
                          trust_repo=True, yoso_version='yoso-normal-v1-5')

# Process each image
for image_path in tqdm.tqdm(image_filenames):
    try:
        # Load image
        input_image = Image.open(image_path)
        
        # Generate normal map
        normal_map = predictor(input_image, data_type=args.data_type)
        
        # Create output path maintaining relative directory structure
        rel_path = image_path.relative_to(input_image_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save the result
        normal_map.save(output_path)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

print("Processing complete!")