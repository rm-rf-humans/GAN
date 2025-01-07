import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

def save_generated_samples(generator, epoch, num_examples, save_dir):
    """Generate and save sample images."""
    noise = tf.random.normal([num_examples, generator.input_shape[1]])
    generated_images = generator(noise, training=False)
    
    # Rescale to [0, 1]
    generated_images = (generated_images + 1) / 2
    
    # Create directory
    save_dir = Path(save_dir)
    epoch_dir = save_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images
    for i in range(num_examples):
        plt.figure(figsize=(8, 2))
        plt.axis('off')
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.savefig(epoch_dir / f"sample_{i+1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

