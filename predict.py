from process_image import process_image
import numpy as np
from PIL import Image
import tensorflow as tf

def predict(image_path, model, top_k=5):

    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)


    expanded_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(expanded_image)

    top_k_values, top_k_indices = tf.nn.top_k(predictions[0], k=top_k)

    return top_k_values.numpy(), top_k_indices.numpy()

