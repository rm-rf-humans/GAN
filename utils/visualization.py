import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Union, List

def save_generated_samples(generator: tf.keras.Model,
                         noise: tf.Tensor,
                         epoch: int,
                         save_dir: Union[str, Path],
                         image_shape: tuple = (32, 128, 1)) -> None:
    """Generate and save sample images.
    
    Args:
        generator: The generator model
        noise: Input noise tensor
        epoch: Current epoch number
        save_dir: Directory to save the generated images
        image_shape: Shape of the generated images
    """
    # Generate images
    generated_images = generator(noise, training=False)
    
    # Rescale to [0, 1]
    generated_images = (generated_images + 1) / 2
    
    # Create directory
    save_dir = Path(save_dir)
    epoch_dir = save_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images
    num_examples = generated_images.shape[0]
    for i in range(num_examples):
        plt.figure(figsize=(8, 2))
        plt.axis('off')
        if image_shape[-1] == 1:  # Grayscale
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        else:  # RGB
            plt.imshow(generated_images[i])
        plt.savefig(epoch_dir / f"sample_{i+1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_training_history(history: Dict[str, List[float]],
                         save_path: Union[str, Path] = None) -> None:
    """Plot training history metrics.
    
    Args:
        history: Dictionary containing lists of metrics
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot each metric in a subplot
    num_metrics = len(history)
    num_cols = min(2, num_metrics)
    num_rows = (num_metrics + num_cols - 1) // num_cols
    
    for idx, (metric_name, values) in enumerate(history.items()):
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.plot(values, label=metric_name)
        plt.title(metric_name.replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def plot_comparison_grid(real_images: np.ndarray,
                        generated_images: np.ndarray,
                        num_samples: int = 5,
                        save_path: Union[str, Path] = None) -> None:
    """Plot a grid comparing real and generated images.
    
    Args:
        real_images: Batch of real images
        generated_images: Batch of generated images
        num_samples: Number of samples to display
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 3))
    
    for i in range(num_samples):
        # Plot real image
        plt.subplot(2, num_samples, i + 1)
        if real_images.shape[-1] == 1:
            plt.imshow(real_images[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(real_images[i])
        plt.axis('off')
        if i == 0:
            plt.title('Real Images')
            
        # Plot generated image
        plt.subplot(2, num_samples, num_samples + i + 1)
        if generated_images.shape[-1] == 1:
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(generated_images[i])
        plt.axis('off')
        if i == 0:
            plt.title('Generated Images')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def plot_interpolation(generator: tf.keras.Model,
                      num_steps: int = 10,
                      noise_dim: int = 100,
                      save_path: Union[str, Path] = None) -> None:
    """Generate and plot interpolation between two random points in latent space.
    
    Args:
        generator: The generator model
        num_steps: Number of interpolation steps
        noise_dim: Dimension of the noise vector
        save_path: Optional path to save the plot
    """
    # Generate two random points in latent space
    z1 = tf.random.normal([1, noise_dim])
    z2 = tf.random.normal([1, noise_dim])
    
    # Create interpolation steps
    alphas = np.linspace(0, 1, num_steps)
    interpolated_images = []
    
    for alpha in alphas:
        z = z1 * (1 - alpha) + z2 * alpha
        interpolated_images.append(generator(z, training=False)[0])
    
    # Plot results
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(interpolated_images):
        plt.subplot(1, num_steps, i + 1)
        if img.shape[-1] == 1:
            plt.imshow(img[:, :, 0], cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')
    
    plt.suptitle('Latent Space Interpolation')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def plot_attention_maps(attention_weights: np.ndarray,
                       original_image: np.ndarray,
                       save_path: Union[str, Path] = None) -> None:
    """Plot attention maps for visualization of model focus areas.
    
    Args:
        attention_weights: Array of attention weights
        original_image: Original input image
        save_path: Optional path to save the plot
    """
    num_attention_heads = attention_weights.shape[-1]
    plt.figure(figsize=(15, 3 * (1 + num_attention_heads // 4)))
    
    # Plot original image
    plt.subplot(num_attention_heads // 4 + 1, 4, 1)
    if original_image.shape[-1] == 1:
        plt.imshow(original_image[:, :, 0], cmap='gray')
    else:
        plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot attention maps
    for i in range(num_attention_heads):
        plt.subplot(num_attention_heads // 4 + 1, 4, i + 2)
        plt.imshow(attention_weights[:, :, i], cmap='viridis')
        plt.title(f'Attention Head {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()
