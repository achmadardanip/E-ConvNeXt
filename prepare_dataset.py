#!/usr/bin/env python3
"""
Helper script to prepare dataset files for E-ConvNeXt training
Supports both classification and detection tasks
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def prepare_classification_dataset(root_dir, output_dir, train_split=0.8):
    """
    Prepare classification dataset in the required format
    
    Expected structure:
    root_dir/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            img2.jpg
    
    Generates:
        train_list.txt
        val_list.txt
    """
    print(f"Preparing classification dataset from {root_dir}")
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: {root_dir} does not exist")
        return
    
    # Get all class directories
    class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    class_dirs.sort()
    
    if not class_dirs:
        print(f"Error: No class directories found in {root_dir}")
        return
    
    print(f"Found {len(class_dirs)} classes:")
    for idx, class_dir in enumerate(class_dirs):
        print(f"  {idx}: {class_dir.name}")
    
    # Collect all images
    all_samples = []
    for class_idx, class_dir in enumerate(class_dirs):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [
            f for f in class_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        for img_path in images:
            # Store relative path from root_dir
            rel_path = img_path.relative_to(root_path)
            all_samples.append((str(rel_path), class_idx))
        
        print(f"  Class {class_idx} ({class_dir.name}): {len(images)} images")
    
    print(f"\nTotal images: {len(all_samples)}")
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * train_split)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write train_list.txt
    train_list_path = output_path / 'train_list.txt'
    with open(train_list_path, 'w') as f:
        for img_path, label in train_samples:
            f.write(f"{img_path} {label}\n")
    
    # Write val_list.txt
    val_list_path = output_path / 'val_list.txt'
    with open(val_list_path, 'w') as f:
        for img_path, label in val_samples:
            f.write(f"{img_path} {label}\n")
    
    # Write class_names.txt for reference
    class_names_path = output_path / 'class_names.txt'
    with open(class_names_path, 'w') as f:
        for idx, class_dir in enumerate(class_dirs):
            f.write(f"{idx} {class_dir.name}\n")
    
    print(f"\n✓ Dataset prepared successfully!")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Output files:")
    print(f"    - {train_list_path}")
    print(f"    - {val_list_path}")
    print(f"    - {class_names_path}")


def prepare_detection_dataset_from_images(images_dir, output_dir, class_name="object"):
    """
    Create a basic COCO format annotation file from images
    (Without actual bounding boxes - for demonstration purposes)
    
    For real object detection, you need to annotate images with tools like:
    - LabelImg
    - CVAT
    - VGG Image Annotator (VIA)
    """
    print(f"Preparing detection dataset from {images_dir}")
    print("Note: This creates dummy annotations for demonstration.")
    print("For real object detection, please annotate your images properly.")
    
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"Error: {images_dir} does not exist")
        return
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        f for f in images_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"Error: No images found in {images_dir}")
        return
    
    print(f"Found {len(images)} images")
    
    # Create COCO format annotation
    coco_annotation = {
        "info": {
            "description": "Auto-generated dataset",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": class_name,
                "supercategory": "object"
            }
        ]
    }
    
    # Add images (without actual annotations)
    for idx, img_path in enumerate(images, 1):
        # Try to get image dimensions
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            width, height = 640, 640  # Default size
        
        coco_annotation["images"].append({
            "id": idx,
            "file_name": img_path.name,
            "height": height,
            "width": width
        })
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write annotation file
    anno_path = output_path / 'instances.json'
    with open(anno_path, 'w') as f:
        json.dump(coco_annotation, f, indent=2)
    
    print(f"\n✓ Detection dataset prepared!")
    print(f"  Images: {len(images)}")
    print(f"  Output: {anno_path}")
    print(f"\n⚠ Warning: This file contains NO bounding box annotations!")
    print(f"  Please annotate your images using tools like LabelImg or CVAT")


def create_sample_config(task, output_dir):
    """Create a sample configuration file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if task == 'classification':
        config = {
            "task": "classification",
            "model_arch": "mini",
            "num_classes": 10,
            "dataset_root": "dataset/custom_dataset",
            "train_list": "dataset/custom_dataset/train_list.txt",
            "val_list": "dataset/custom_dataset/val_list.txt",
            "image_size": 224,
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.0001
        }
    else:  # detection
        config = {
            "task": "detection",
            "model_arch": "mini",
            "num_classes": 4,
            "dataset_root": "dataset/coco_format",
            "train_anno": "dataset/coco_format/annotations/instances_train.json",
            "val_anno": "dataset/coco_format/annotations/instances_val.json",
            "image_size": 640,
            "batch_size": 8,
            "epochs": 50,
            "learning_rate": 0.0001
        }
    
    config_path = output_path / f'config_{task}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Sample configuration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for E-ConvNeXt training')
    parser.add_argument('--task', choices=['classification', 'detection'], required=True,
                      help='Task type: classification or detection')
    parser.add_argument('--root-dir', required=True,
                      help='Root directory of the dataset')
    parser.add_argument('--output-dir', default='dataset',
                      help='Output directory for processed files')
    parser.add_argument('--train-split', type=float, default=0.8,
                      help='Training split ratio (default: 0.8)')
    parser.add_argument('--class-name', default='object',
                      help='Class name for detection (default: object)')
    parser.add_argument('--create-config', action='store_true',
                      help='Create sample configuration file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("E-ConvNeXt Dataset Preparation Tool")
    print("=" * 60)
    
    if args.task == 'classification':
        prepare_classification_dataset(args.root_dir, args.output_dir, args.train_split)
    else:  # detection
        prepare_detection_dataset_from_images(args.root_dir, args.output_dir, args.class_name)
    
    if args.create_config:
        create_sample_config(args.task, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Review the generated files")
    print("2. Update the notebook configuration (Section 2)")
    print("3. Run the training notebook")
    print("=" * 60)


if __name__ == '__main__':
    main()
