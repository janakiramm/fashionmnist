import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import argparse

tf.logging.set_verbosity(tf.logging.ERROR)

#Parse input parameters
parser = argparse.ArgumentParser(description='Fashion MNIST Keras Model')
parser.add_argument('--modelPath', type=str, dest='MODEL_DIR', help='location to store the model artifacts')
parser.add_argument('--version', type=str, dest='VERSION', default="1", help='model version')
args = parser.parse_args()

MODEL_DIR = args.MODEL_DIR
VERSION = args.VERSION

#Download Fashion MNIST dataset and split it for train and test
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Scale images
train_images = train_images / 255.0
test_images = test_images / 255.0

#Reshape array
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#Build Keras model
model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
])

model.summary()

#Read the EPOCH value from environment variable
epochs = int(os.getenv("EPOCHS",1))

#Compile and fit
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs)

#Check accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nModel accuracy: {}'.format(test_acc))

#Save model 
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    export_path = os.path.join(MODEL_DIR, VERSION)
    print('export_path = {}\n'.format(export_path))

    tf.saved_model.simple_save(
        keras.backend.get_session(),
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name:t for t in model.outputs})

    print('\nModel saved to ' + MODEL_DIR)
else:
    print('\nExisting model found at ' + MODEL_DIR)
    print('\nDid not overwrite old model. Run the job again with a different location to store the model')