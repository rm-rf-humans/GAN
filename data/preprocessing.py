import tensorflow as tf

def preprocess_image(img_path, img_height, img_width):
    """Load and preprocess a single image."""
    try:
        img = tf.keras.utils.load_img(img_path, color_mode="grayscale")
        img = tf.keras.utils.img_to_array(img)
        img = tf.image.resize(img, [img_height, img_width])
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        return img
    except Exception as e:
        raise Exception(f"Failed to preprocess image {img_path}: {str(e)}")

