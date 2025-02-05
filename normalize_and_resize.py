import tensorflow as tf

def normalize_and_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0
    return image, label
