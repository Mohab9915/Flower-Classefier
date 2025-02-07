import tensorflow as tf

def process_image(image):
    """
    Process an image for model prediction:
    1. Convert to tensor
    2. Resize to 224x224 (MobileNetV2 input size)
    3. Normalize pixel values to [0,1]
    
    Args:
        image: numpy array of image data
    Returns:
        processed image as numpy array
    """
    # Convert image to tensor format
    image = tf.convert_to_tensor(image)
    
    # Resize image to match model's expected input
    image = tf.image.resize(image, (224, 224))
    
    # Normalize pixel values to [0,1]
    image = image / 255.0
    return image.numpy()