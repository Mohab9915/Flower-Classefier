import tensorflow_datasets as tfds
import tensorflow as tf
import json
import tensorflow_hub as hub
import tf_keras
from normalize_and_resize import normalize_and_resize

dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
train_set, validation_set, test_set = dataset['train'], dataset['validation'], dataset['test']

num_train = dataset_info.splits['train'].num_examples
num_val = dataset_info.splits['validation'].num_examples
num_test = dataset_info.splits['test'].num_examples
print(f'Training samples: {num_train}, Validation samples: {num_val}, Test samples: {num_test}')

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

BATCH_SIZE = 32
train_batches = (train_set
                 .shuffle(1000)
                 .map(normalize_and_resize)
                 .batch(BATCH_SIZE)
                 .prefetch(1))

validation_batches = (validation_set
                     .map(normalize_and_resize)
                     .batch(BATCH_SIZE)
                     .prefetch(1))

test_batches = (test_set
                .map(normalize_and_resize)
                .batch(BATCH_SIZE)
                .prefetch(1))

model = tf_keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/100-224-feature-vector/2",
                   trainable=False), 
    tf_keras.layers.Dense(512, activation='softmax'),
    tf_keras.layers.Dropout(0.2),
    tf_keras.layers.Dense(102, activation='softmax')
])

model.build([None, 224, 224, 3])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

NUM_EPOCHS = 15
history = model.fit(
    train_batches,
    epochs=NUM_EPOCHS,
    validation_data=validation_batches
)

model.save('flower_classifier_model.h5')
