import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Dict, Union
import logging

class GANMetrics:
    """Class for computing and tracking various GAN evaluation metrics."""
    
    def __init__(self, real_image_shape: Tuple[int, ...]):
        """
        Initialize the metrics calculator.
        
        Args:
            real_image_shape: The shape of real images (height, width, channels)
        """
        self.real_image_shape = real_image_shape
        self.metrics_history = {
            'fid_scores': [],
            'inception_scores': [],
            'mmd_scores': [],
            'wasserstein_distances': []
        }
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('GANMetrics')
    
    def compute_inception_score(self, generated_images: np.ndarray, n_split: int = 10) -> float:
        """
        Compute the Inception Score for generated images.
        
        Args:
            generated_images: Batch of generated images
            n_split: Number of splits for computing the score
            
        Returns:
            float: Inception score
        """
        try:
            # Ensure images are in correct format
            if generated_images.shape[1:] != self.real_image_shape:
                raise ValueError(f"Generated images shape {generated_images.shape[1:]} "
                               f"doesn't match expected shape {self.real_image_shape}")
            
            # Normalize images if needed
            if generated_images.max() > 1.0:
                generated_images = generated_images / 255.0
            
            # Split images into n_split groups
            split_scores = []
            n_images = generated_images.shape[0]
            split_size = n_images // n_split
            
            for i in range(n_split):
                split = generated_images[i * split_size:(i + 1) * split_size]
                # Compute split score (simplified version without actual Inception model)
                # In practice, you would use a pre-trained Inception model here
                split_score = np.std(split) + np.mean(split)  # Simplified metric
                split_scores.append(split_score)
            
            inception_score = np.mean(split_scores)
            self.metrics_history['inception_scores'].append(inception_score)
            return inception_score
            
        except Exception as e:
            self.logger.error(f"Error computing Inception Score: {str(e)}")
            return 0.0
    
    def compute_fid_score(self, real_features: np.ndarray, 
                         generated_features: np.ndarray) -> float:
        """
        Compute Fréchet Inception Distance between real and generated images.
        
        Args:
            real_features: Features extracted from real images
            generated_features: Features extracted from generated images
            
        Returns:
            float: FID score
        """
        try:
            # Calculate mean and covariance statistics
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
            
            # Calculate sum squared difference between means
            diff = mu1 - mu2
            
            # Calculate sqrt of product between cov
            covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma1, sigma2))
            
            # Check and correct imaginary numbers from sqrt
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            # Calculate score
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
            
            self.metrics_history['fid_scores'].append(fid)
            return float(fid)
            
        except Exception as e:
            self.logger.error(f"Error computing FID score: {str(e)}")
            return float('inf')
    
    def compute_mmd_score(self, real_samples: np.ndarray, 
                         generated_samples: np.ndarray,
                         kernel='rbf') -> float:
        """
        Compute Maximum Mean Discrepancy between real and generated samples.
        
        Args:
            real_samples: Samples from real distribution
            generated_samples: Samples from generated distribution
            kernel: Kernel type for MMD computation
            
        Returns:
            float: MMD score
        """
        try:
            # Flatten samples
            real_samples = real_samples.reshape(real_samples.shape[0], -1)
            generated_samples = generated_samples.reshape(generated_samples.shape[0], -1)
            
            # Compute kernel matrices
            K_XX = pairwise_distances(real_samples, metric='euclidean')
            K_YY = pairwise_distances(generated_samples, metric='euclidean')
            K_XY = pairwise_distances(real_samples, generated_samples, metric='euclidean')
            
            # Apply RBF kernel
            gamma = 1.0 / real_samples.shape[1]
            K_XX = np.exp(-gamma * K_XX)
            K_YY = np.exp(-gamma * K_YY)
            K_XY = np.exp(-gamma * K_XY)
            
            # Compute MMD
            mmd = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()) ** 0.5
            
            self.metrics_history['mmd_scores'].append(mmd)
            return float(mmd)
            
        except Exception as e:
            self.logger.error(f"Error computing MMD score: {str(e)}")
            return float('inf')
    
    def compute_wasserstein_distance(self, real_samples: np.ndarray, 
                                   generated_samples: np.ndarray) -> float:
        """
        Compute Wasserstein distance between real and generated distributions.
        
        Args:
            real_samples: Samples from real distribution
            generated_samples: Samples from generated distribution
            
        Returns:
            float: Wasserstein distance
        """
        try:
            # Sort the samples
            real_samples = np.sort(real_samples.flatten())
            generated_samples = np.sort(generated_samples.flatten())
            
            # Compute Wasserstein distance
            n = len(real_samples)
            m = len(generated_samples)
            
            # Interpolate if necessary
            if n != m:
                x = np.linspace(0, 1, n)
                y = np.linspace(0, 1, m)
                real_interp = np.interp(x, np.linspace(0, 1, n), real_samples)
                gen_interp = np.interp(x, np.linspace(0, 1, m), generated_samples)
            else:
                real_interp = real_samples
                gen_interp = generated_samples
            
            # Calculate distance
            wasserstein_dist = np.abs(real_interp - gen_interp).mean()
            
            self.metrics_history['wasserstein_distances'].append(wasserstein_dist)
            return float(wasserstein_dist)
            
        except Exception as e:
            self.logger.error(f"Error computing Wasserstein distance: {str(e)}")
            return float('inf')
    
    def plot_metrics_history(self, save_path: Union[str, Path] = None):
        """
        Plot the history of all tracked metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('GAN Evaluation Metrics History')
            
            # Plot each metric
            for (metric, values), ax in zip(
                self.metrics_history.items(),
                axes.flat
            ):
                ax.plot(values, label=metric)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Score')
                ax.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Metrics plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting metrics history: {str(e)}")
    
    def save_metrics(self, save_path: Union[str, Path]):
        """
        Save metrics history to a file.
        
        Args:
            save_path: Path to save the metrics
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(save_path, self.metrics_history)
            self.logger.info(f"Metrics history saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
    
    def load_metrics(self, load_path: Union[str, Path]):
        """
        Load metrics history from a file.
        
        Args:
            load_path: Path to load the metrics from
        """
        try:
            load_path = Path(load_path)
            self.metrics_history = np.load(load_path, allow_pickle=True).item()
            self.logger.info(f"Metrics history loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading metrics: {str(e)}")
