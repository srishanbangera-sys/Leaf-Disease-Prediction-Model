import tensorflow as tf
from tensorflow import keras'
from tensorflow.keras import layers
import os
import json

DATASET_PATH = "dataset/PlantVillage"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10


if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset was not found.")
    exit()

train_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,
    image_size = IMAGE_SIZE,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    shuffle = True

)

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,
    image_size = IMAGE_SIZE,
    validation_split = 0.2,
    subset = "validation",
    seed = 123
    shuffle = False

)





