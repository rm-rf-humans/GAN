import tensorflow as tf
from tensorflow.keras import layers, models
from config.config import Config

def build_gan(generator, discriminator):
    discriminator.trainable = False
    
    gan_input = layers.Input(shape=(Config.NOISE_DIM,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    
    model = models.Model(gan_input, gan_output)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=Config.LEARNING_RATE,
        beta_1=Config.BETA_1,
        beta_2=Config.BETA_2
    )
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    return model

