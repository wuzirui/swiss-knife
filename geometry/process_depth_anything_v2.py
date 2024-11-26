"""
Depth Map Generator using Depth Anything V2

This script processes a directory of images to generate depth maps using the Depth Anything V2 model.

Requirements:
    pip install torch torchvision
    pip install pillow tqdm
    pip install transformers
    
Usage:
    python process_depth_anything_v2.py --input_image_dir path/to/input/images \
                                      --output_image_dir path/to/output/depths \
                                      --model_size small

Example:
    python process_depth_anything_v2.py --input_image_dir ./photos \
                                      --output_image_dir ./depth_maps \
                                      --model_size base
"""

import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DepthAnythingConfig, DepthAnythingForDepthEstimation

def get_model_name(size):
    """Get the model name based on size parameter."""
    size_mapping = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf'
    }
    return size_mapping.get(size, size_mapping['small'])

def get_model_config(size, depth_type="relative", max_depth=None):
    """Get model configuration based on size and depth estimation type."""
    
    config = DepthAnythingConfig(
        depth_estimation_type=depth_type,
        max_depth=max_depth,
    )
    return config

class DepthAnythingPredictor:
    def __init__(self, model_name, depth_type="relative", max_depth=None, 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Depth Anything V2 model: {model_name}")
        
        # Get model size from name
        size = model_name.split('-')[-2].lower()
        
        # Create custom configuration
        config = get_model_config(size, depth_type, max_depth)
        
        # Load the image processor and model with pretrained weights
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_name,  # This loads the pretrained weights
        ).to(device)
        
        # Update the model config if needed
        if depth_type != "relative" or max_depth is not None:
            self.model.config.depth_estimation_type = depth_type
            self.model.config.max_depth = max_depth
            
        self.model.eval()

    def __call__(self, image):
        """
        Process an image and return the depth map.
        
        Args:
            image: PIL Image
            
        Returns:
            depth_map: PIL Image of depth values
        """
        # Prepare image for the model
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process the output
        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        # Get and normalize the predicted depth map
        predicted_depth = post_processed_output[0]["predicted_depth"]
        
        if self.model.config.depth_estimation_type == "metric":
            # For metric depth, convert to cm and scale to [0, 255]
            depth = predicted_depth.detach().cpu().numpy() * 100  # meters to cm
            depth = np.clip(depth, 0, 65535)  # clip to uint16 range
            depth_map = Image.fromarray(depth.astype("uint16"))
        else:
            # For relative depth, normalize to [0, 255]
            depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
            depth = depth.detach().cpu().numpy() * 255
            depth_map = Image.fromarray(depth.astype("uint8"))
            
        return depth_map

def main():
    parser = argparse.ArgumentParser(description="Generate depth maps from images using Depth Anything V2")
    parser.add_argument("--input_image_dir", type=str, required=True,
                        help="Directory containing input images (will search recursively)")
    parser.add_argument("--output_image_dir", type=str, required=True,
                        help="Directory where depth maps will be saved")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["small", "base", "large"],
                        help="Size of the Depth Anything V2 model to use")
    parser.add_argument("--depth_type", type=str, default="relative",
                        choices=["relative", "metric"],
                        help="Type of depth estimation (relative or metric)")
    parser.add_argument("--max_depth", type=float, default=None,
                        help="Maximum depth value (use 20 for indoor, 80 for outdoor when using metric depth)")

    args = parser.parse_args()
    print(args)

    # Setup paths
    input_image_dir = Path(args.input_image_dir)
    output_dir = Path(args.output_image_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get list of images
    image_filenames = sorted(list(input_image_dir.glob("**/*.png")))
    image_filenames += sorted(list(input_image_dir.glob("**/*.jpg")))
    print(f"Found {len(image_filenames)} images to process.")

    # Create predictor instance with updated parameters
    model_name = get_model_name(args.model_size)
    predictor = DepthAnythingPredictor(
        model_name,
        depth_type=args.depth_type,
        max_depth=args.max_depth
    )

    # Process each image
    for image_path in tqdm.tqdm(image_filenames):
        try:
            # Load image
            input_image = Image.open(image_path).convert('RGB')
            
            # Generate depth map
            depth_map = predictor(input_image)
            
            # Create output path maintaining relative directory structure
            rel_path = image_path.relative_to(input_image_dir)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Save the result
            depth_map.save(output_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    print("Processing complete!")

if __name__ == "__main__":
    main()
