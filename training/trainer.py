import tensorflow as tf
from pathlib import Path
import time
import numpy as np
from config.config import Config
from .callbacks import setup_callbacks
from utils.metrics import GANMetrics

class GANTrainer:
    def __init__(self, generator, discriminator, gan, dataset):
        self.generator = generator
        self.discriminator = discriminator
        self.gan = gan
        self.dataset = dataset
        self.callback_handler = setup_callbacks(generator, discriminator)
        self.metrics = GANMetrics(real_image_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS))
        
        # Create fixed noise vector for consistent metric evaluation
        self.fixed_noise = tf.random.normal([100, Config.NOISE_DIM])
        
        # Initialize metric tracking
        self.best_inception_score = float('-inf')
        self.best_fid_score = float('inf')
        
    def train_step(self, batch_images):
        batch_size = tf.shape(batch_images)[0]
        
        # Generate noise
        noise = tf.random.normal([batch_size, Config.NOISE_DIM])
        
        # Generate fake images
        generated_images = self.generator(noise, training=True)
        
        # Labels
        real_labels = tf.ones((batch_size, 1)) * 0.9  # Label smoothing
        fake_labels = tf.zeros((batch_size, 1))
        
        # Train discriminator
        self.discriminator.trainable = True
        d_loss_real = self.discriminator.train_on_batch(batch_images, real_labels)
        d_loss_fake = self.discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Train generator
        self.discriminator.trainable = False
        noise = tf.random.normal([batch_size, Config.NOISE_DIM])
        g_loss = self.gan.train_on_batch(noise, real_labels)
        
        return d_loss, g_loss, generated_images
    
    def compute_epoch_metrics(self, real_images, epoch):
        """
        Compute various GAN metrics for the current epoch.
        
        Args:
            real_images: Batch of real images for comparison
            epoch: Current epoch number
            
        Returns:
            dict: Dictionary containing computed metrics
        """
        # Generate a fixed set of images for consistent evaluation
        generated_images = self.generator(self.fixed_noise, training=False)
        generated_images = tf.cast(generated_images, tf.float32)
        
        # Normalize images to [0, 1] if needed
        if tf.reduce_max(generated_images) > 1.0:
            generated_images = (generated_images + 1.0) / 2.0
        if tf.reduce_max(real_images) > 1.0:
            real_images = (real_images + 1.0) / 2.0
            
        # Convert to numpy for metric computation
        generated_np = generated_images.numpy()
        real_np = real_images.numpy()
        
        metrics_dict = {}
        
        # Compute Inception Score
        inception_score = self.metrics.compute_inception_score(generated_np)
        metrics_dict['inception_score'] = inception_score
        
        # Extract features for FID (using discriminator's intermediate layer as feature extractor)
        real_features = self.get_features(real_np)
        generated_features = self.get_features(generated_np)
        
        # Compute FID Score
        fid_score = self.metrics.compute_fid_score(real_features, generated_features)
        metrics_dict['fid_score'] = fid_score
        
        # Compute MMD Score
        mmd_score = self.metrics.compute_mmd_score(real_np, generated_np)
        metrics_dict['mmd_score'] = mmd_score
        
        # Compute Wasserstein Distance
        wasserstein_dist = self.metrics.compute_wasserstein_distance(real_np, generated_np)
        metrics_dict['wasserstein_distance'] = wasserstein_dist
        
        # Update best scores
        if inception_score > self.best_inception_score:
            self.best_inception_score = inception_score
            self.save_best_model('inception', epoch)
            
        if fid_score < self.best_fid_score:
            self.best_fid_score = fid_score
            self.save_best_model('fid', epoch)
            
        return metrics_dict
    
    def get_features(self, images):
        """
        Extract features from the discriminator's intermediate layer.
        
        Args:
            images: Batch of images
            
        Returns:
            numpy.ndarray: Extracted features
        """
        # Create a feature extractor model using an intermediate layer
        feature_layer = self.discriminator.layers[-2].output  # Use the layer before the final dense layer
        feature_extractor = tf.keras.Model(self.discriminator.input, feature_layer)
        
        # Extract features
        features = feature_extractor.predict(images)
        return features
    
    def save_best_model(self, metric_name, epoch):
        """
        Save the model when it achieves the best metric score.
        
        Args:
            metric_name: Name of the metric (e.g., 'inception' or 'fid')
            epoch: Current epoch number
        """
        save_dir = Config.CHECKPOINTS_DIR / f"best_{metric_name}_score_epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator.save(save_dir / "generator.h5")
        self.discriminator.save(save_dir / "discriminator.h5")
        
        print(f"\nSaved best {metric_name} score model at epoch {epoch}")
    
    def train(self, epochs):
        """
        Train the GAN with comprehensive metric tracking and visualization.
        
        Args:
            epochs: Number of epochs to train
        """
        self.callback_handler.on_training_begin()
        start_time = time.time()
        
        for epoch in range(epochs):
            self.callback_handler.on_epoch_begin(epoch)
            epoch_start_time = time.time()
            
            # Training loop
            d_losses = []
            g_losses = []
            real_images_batch = None
            
            for batch_idx, (batch_images, _) in enumerate(self.dataset):
                self.callback_handler.on_batch_begin(batch_idx)
                
                # Store a batch of real images for metric computation
                if real_images_batch is None:
                    real_images_batch = batch_images
                
                d_loss, g_loss, generated_images = self.train_step(batch_images)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                
                self.callback_handler.on_batch_end(batch_idx, {
                    'd_loss': d_loss,
                    'g_loss': g_loss
                })
            
            # Calculate epoch losses
            epoch_d_loss = tf.reduce_mean(d_losses)
            epoch_g_loss = tf.reduce_mean(g_losses)
            
            # Compute metrics
            metrics_dict = self.compute_epoch_metrics(real_images_batch, epoch)
            
            # Update callback logs
            logs = {
                'd_loss': epoch_d_loss,
                'g_loss': epoch_g_loss,
                'generator': self.generator,
                'discriminator': self.discriminator,
                'generated_images': generated_images,
                **metrics_dict  # Include all computed metrics
            }
            
            # Calculate epoch time and ETA
            epoch_time = time.time() - epoch_start_time
            eta = (epochs - epoch - 1) * epoch_time
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f}")
            print(f"Inception Score: {metrics_dict['inception_score']:.4f}")
            print(f"FID Score: {metrics_dict['fid_score']:.4f}")
            print(f"MMD Score: {metrics_dict['mmd_score']:.4f}")
            print(f"Wasserstein Distance: {metrics_dict['wasserstein_distance']:.4f}")
            print(f"Time: {epoch_time:.2f}s | ETA: {eta/3600:.2f}h")
            
            self.callback_handler.on_epoch_end(epoch, logs)
            
            # Save metrics plot
            if (epoch + 1) % 10 == 0:
                self.metrics.plot_metrics_history(Config.METRICS_DIR / f"metrics_plot_epoch_{epoch+1}.png")
            
            # Check for early stopping
            if any(callback.stop_training for callback in 
                  self.callback_handler.callbacks if 
                  hasattr(callback, 'stop_training') and 
                  callback.stop_training):
                break
        
        # End of training
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best Inception Score: {self.best_inception_score:.4f}")
        print(f"Best FID Score: {self.best_fid_score:.4f}")
        
        # Save final metrics
        self.metrics.save_metrics(Config.METRICS_DIR / "final_metrics.npy")
        self.metrics.plot_metrics_history(Config.METRICS_DIR / "final_metrics_plot.png")
        
        self.callback_handler.on_training_end()
