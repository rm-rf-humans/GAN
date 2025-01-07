import tensorflow as tf
from data.dataset import IAMDataset
from models.generator import build_generator
from models.discriminator import build_discriminator
from models.gan import build_gan
from training.trainer import GANTrainer
from config.config import Config

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = IAMDataset(Config.IAM_WORDS_DIR, Config.IAM_ASCII_FILE)
    images, labels = dataset.load_dataset()
    train_dataset = dataset.create_tf_dataset(images, labels)
    
    # Build models
    print("Building models...")
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    # Print model summaries
    generator.summary()
    discriminator.summary()
    
    # Initialize trainer
    trainer = GANTrainer(generator, discriminator, gan, train_dataset)
    
    # Train
    print("Starting training...")
    trainer.train(Config.EPOCHS)

if __name__ == "__main__":
    main()

