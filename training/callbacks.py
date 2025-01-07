# training/callbacks.py
import tensorflow as tf
import time
from pathlib import Path
import numpy as np
from datetime import datetime
from config.config import Config
from utils.visualization import save_generated_samples, plot_training_history

class GANCallbackHandler:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.callbacks = []
        self._init_time = time.time()
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'epoch_times': []
        }

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def on_training_begin(self):
        for callback in self.callbacks:
            callback.on_training_begin()

    def on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def update_history(self, d_loss, g_loss):
        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss)
        self.history['epoch_times'].append(time.time() - self._init_time)

class BaseCallback:
    def on_training_begin(self): pass
    def on_training_end(self): pass
    def on_epoch_begin(self, epoch): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_begin(self, batch): pass
    def on_batch_end(self, batch, logs=None): pass

class ModelCheckpoint(BaseCallback):
    def __init__(self, checkpoint_dir, save_freq=10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save models
            logs['generator'].save(save_dir / "generator.h5")
            logs['discriminator'].save(save_dir / "discriminator.h5")
            
            print(f"\nSaved checkpoint for epoch {epoch+1} at {save_dir}")

class SampleGenerator(BaseCallback):
    def __init__(self, sample_dir, num_samples=10, save_freq=10):
        self.sample_dir = Path(sample_dir)
        self.num_samples = num_samples
        self.save_freq = save_freq
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self._fixed_noise = None

    def on_training_begin(self):
        # Generate fixed noise for consistent sampling
        self._fixed_noise = tf.random.normal([self.num_samples, Config.NOISE_DIM])

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            save_generated_samples(
                generator=logs['generator'],
                noise=self._fixed_noise,
                epoch=epoch + 1,
                save_dir=self.sample_dir
            )

class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = self.log_dir / f"gan_{current_time}"
        self.writer = tf.summary.create_file_writer(str(self.train_log_dir))

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            tf.summary.scalar('discriminator_loss', logs['d_loss'], step=epoch)
            tf.summary.scalar('generator_loss', logs['g_loss'], step=epoch)
            
            # Add generated images to TensorBoard
            if 'generated_images' in logs:
                tf.summary.image(
                    'generated_images',
                    logs['generated_images'],
                    step=epoch,
                    max_outputs=5
                )

class ProgressLogger(BaseCallback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self._epoch_start_time = None

    def on_epoch_begin(self, epoch):
        self._epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{self.total_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self._epoch_start_time
        print(f"D Loss: {logs['d_loss']:.4f}")
        print(f"G Loss: {logs['g_loss']:.4f}")
        print(f"Time: {epoch_time:.2f} sec")

class EarlyStopping(BaseCallback):
    def __init__(self, monitor='g_loss', patience=10, min_delta=0.01):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = None
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current = logs[self.monitor]
        
        if self.best is None:
            self.best = current
        elif current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                print(f"\nEarly stopping triggered at epoch {epoch+1}")

# Example usage in training/trainer.py
def setup_callbacks(generator, discriminator):
    """Set up training callbacks."""
    handler = GANCallbackHandler(generator, discriminator)
    
    # Add callbacks
    handler.add_callback(ModelCheckpoint(
        checkpoint_dir=Config.CHECKPOINTS_DIR,
        save_freq=10
    ))
    
    handler.add_callback(SampleGenerator(
        sample_dir=Config.SAMPLES_DIR,
        num_samples=Config.NUM_EXAMPLES,
        save_freq=Config.SAMPLE_INTERVAL
    ))
    
    handler.add_callback(TensorBoardCallback(
        log_dir=Config.BASE_DIR / "logs"
    ))
    
    handler.add_callback(ProgressLogger(
        total_epochs=Config.EPOCHS
    ))
    
    handler.add_callback(EarlyStopping(
        monitor='g_loss',
        patience=15
    ))
    
    return handler
