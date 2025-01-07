import tensorflow as tf
from tensorflow.keras import layers, models
from config.config import Config

def build_generator():
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(Config.NOISE_DIM,)),
        
        # Dense and reshape
        layers.Dense(8 * 32 * 64),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((8, 32, 64)),
        
        # Upsampling blocks
        layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # Output layer
        layers.Conv2D(Config.CHANNELS, (4, 4), padding='same', activation='tanh')
    ])
    
    return model

