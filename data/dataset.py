import tensorflow as tf
import numpy as np
from pathlib import Path
from .preprocessing import preprocess_image
from config.config import Config

class IAMDataset:
    def __init__(self, words_dir, ascii_file):
        self.words_dir = Path(words_dir)
        self.ascii_file = Path(ascii_file)
        self.validate_paths()
        
    def validate_paths(self):
        if not self.words_dir.exists():
            raise FileNotFoundError(f"Words directory not found: {self.words_dir}")
        if not self.ascii_file.exists():
            raise FileNotFoundError(f"ASCII file not found: {self.ascii_file}")
    
    def load_dataset(self):
        """Load and parse the IAM dataset."""
        images = []
        labels = []
        processed = 0
        
        with open(self.ascii_file, "r", encoding='utf-8') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                
                try:
                    parts = line.strip().split()
                    word_id = parts[0]
                    text = " ".join(parts[8:])
                    
                    form_id = word_id.split("-")[0]
                    subform_id = f"{form_id}-{word_id.split('-')[1]}"
                    image_path = self.words_dir / form_id / subform_id / f"{word_id}.png"
                    
                    if image_path.exists():
                        img = preprocess_image(str(image_path), Config.IMG_HEIGHT, Config.IMG_WIDTH)
                        images.append(img)
                        labels.append(text)
                        processed += 1
                        
                        if processed % 1000 == 0:
                            print(f"Processed {processed} images")
                            
                except Exception as e:
                    print(f"Error processing {word_id}: {str(e)}")
                    continue
        
        return np.array(images), np.array(labels)
    
    def create_tf_dataset(self, images, labels):
        """Create a TensorFlow dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(len(images))
        dataset = dataset.batch(Config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

