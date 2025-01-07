import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path



IAM_WORDS_DIR = "iam/words"
IAM_ASCII_FILE = "iam/ascii/words.txt"



def parse_iam_dataset(words_dir, ascii_file):
    """
    Parse the IAM word-level dataset with correct directory structure handling
    Directory structure: words_dir/r06/r06-143/r06-143-04-10.png
    """
    images = []
    labels = []
    skipped = 0
    processed = 0
    
    # Convert to Path objects for better path handling
    words_dir = Path(words_dir)
    ascii_file = Path(ascii_file)
    
    # Verify paths exist
    if not words_dir.exists():
        raise FileNotFoundError(f"Words directory not found: {words_dir}")
    if not ascii_file.exists():
        raise FileNotFoundError(f"ASCII file not found: {ascii_file}")
    
    print(f"Starting to parse dataset from {ascii_file}")
    
    with open(ascii_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        
        # Count total valid lines for progress tracking
        total_lines = sum(1 for line in lines if not line.startswith("#") and line.strip())
        print(f"Found {total_lines} total entries in ASCII file")
        
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
                
            try:
                # Split line and handle potential formatting issues
                parts = line.strip().split()
                if len(parts) < 9:
                    print(f"Skipping malformed line: {line.strip()}")
                    skipped += 1
                    continue
                
                # Extract word ID and text
                word_id = parts[0]  # e.g., "r06-143-04-10"
                text = " ".join(parts[8:])
                
                # Parse the word ID to get directory structure
                # word_id format: r06-143-04-10
                word_components = word_id.split("-")
                if len(word_components) < 2:
                    print(f"Invalid word ID format: {word_id}")
                    skipped += 1
                    continue
                
                # Create path: words_dir/r06/r06-143/r06-143-04-10.png
                form_id = word_components[0]  # r06
                subform_id = f"{form_id}-{word_components[1]}"  # r06-143
                image_path = words_dir / form_id / subform_id / f"{word_id}.png"
                
                # Check if image exists
                if not image_path.exists():
                    print(f"Image not found: {image_path}")
                    skipped += 1
                    continue
                
                # Load and preprocess image
                try:
                    img = preprocess_image(str(image_path))
                    images.append(img)
                    labels.append(text)
                    processed += 1
                    
                    # Print progress every 1000 samples
                    if processed % 1000 == 0:
                        print(f"Processed {processed} images...")
                        
                except Exception as e:
                    print(f"Error processing image {image_path}: {str(e)}")
                    skipped += 1
                    continue
                    
            except Exception as e:
                print(f"Error parsing line: {str(e)}")
                skipped += 1
                continue
    
    # Convert to numpy arrays
    images_array = np.array(images) if images else np.array([])
    labels_array = np.array(labels) if labels else np.array([])
    
    print(f"\nDataset parsing complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Skipped: {skipped} entries")
    print(f"Final dataset size: {len(images_array)} images")
    
    if len(images_array) == 0:
        raise ValueError("No valid images were found in the dataset")
        
    return images_array, labels_array

def preprocess_image(img_path):
    try:
        img = tf.keras.utils.load_img(img_path, color_mode="grayscale")
        img = tf.keras.utils.img_to_array(img)
        img = tf.image.resize(img, [32, 128])
        img = img / 255.0
        return img
    except Exception as e:
        raise Exception(f"Failed to preprocess image {img_path}: {str(e)}")

def test_dataset_loading(words_dir, ascii_file):
    try:
        print("Testing dataset loading...")
        x_data, y_data = parse_iam_dataset(words_dir, ascii_file)
        
        print("\nDataset Statistics:")
        print(f"Number of samples: {len(x_data)}")
        print(f"Image shape: {x_data[0].shape}")
        print(f"Sample text: {y_data[0]}")
        
        return x_data, y_data
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None


x_data, y_data = test_dataset_loading(IAM_WORDS_DIR, IAM_ASCII_FILE)
