import tensorflow as tf

def process_image(image):

    image = tf.convert_to_tensor(image)

    image = tf.image.resize(image, (224, 224))

    image = image / 255.0
    return image.numpy()