from process_image import process_image
import numpy as np
from PIL import Image
import tensorflow as tf

def predict(image_path, model, top_k=5):
    """
    Make predictions on an image using the trained model
    
    Args:
        image_path: path to the image file
        model: loaded tensorflow model
        top_k: number of top predictions to return
    Returns:
        tuple of (probabilities, class indices) for top k predictions
    """
    # Load and convert image to numpy array
    im = Image.open(image_path)
    image = np.asarray(im)
    
    # Process image for model input
    processed_image = process_image(image)
    
    # Add batch dimension required by model
    expanded_image = np.expand_dims(processed_image, axis=0)
    
    # Get model predictions
    predictions = model.predict(expanded_image)
    
    # Get top k predictions
    top_k_values, top_k_indices = tf.nn.top_k(predictions[0], k=top_k)
    
    return top_k_values.numpy(), top_k_indices.numpy()

