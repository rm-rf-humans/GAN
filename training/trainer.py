# training/trainer.py
import tensorflow as tf
from pathlib import Path
import time
from config.config import Config
from .callbacks import setup_callbacks

class GANTrainer:
    def __init__(self, generator, discriminator, gan, dataset):
        self.generator = generator
        self.discriminator = discriminator
        self.gan = gan
        self.dataset = dataset
        self.callback_handler = setup_callbacks(generator, discriminator)
        
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
    
    def train(self, epochs):
        self.callback_handler.on_training_begin()
        
        for epoch in range(epochs):
            self.callback_handler.on_epoch_begin(epoch)
            
            # Training loop
            d_losses = []
            g_losses = []
            
            for batch_idx, (batch_images, _) in enumerate(self.dataset):
                self.callback_handler.on_batch_begin(batch_idx)
                
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
            
            # Update callback logs
            logs = {
                'd_loss': epoch_d_loss,
                'g_loss': epoch_g_loss,
                'generator': self.generator,
                'discriminator': self.discriminator,
                'generated_images': generated_images
            }
            
            self.callback_handler.on_epoch_end(epoch, logs)
            
            # Check for early stopping
            if any(callback.stop_training for callback in 
                  self.callback_handler.callbacks if 
                  hasattr(callback, 'stop_training') and 
                  callback.stop_training):
                break
        
        self.callback_handler.on_training_end()
