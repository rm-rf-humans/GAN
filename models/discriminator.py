import tensorflow as tf
from tensorflow.keras import layers, models
from config.config import Config

def build_discriminator():
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS)),
        
        # Convolutional blocks
        layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        # Output
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

