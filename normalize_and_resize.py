import tensorflow as tf

def normalize_and_resize(image, label):
    """
    Preprocessing function for dataset images
    
    Args:
        image: input image tensor
        label: corresponding class label
    Returns:
        tuple of (processed_image, label)
    """
    # Convert image to float32
    image = tf.cast(image, tf.float32)
    
    # Resize to model input size
    image = tf.image.resize(image, [224, 224])
    
    # Normalize pixel values
    image /= 255.0
    return image, label
