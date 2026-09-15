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

class_name = train_dataset.class_names

for index, class_name in enumerate(class_name):

    print(
        index,
        ":",
        class_name
    )

    num_classes = len(class_name)

    if num_classes < 2:
        print("Error: Only on class was detected")

        print("Check your dataset folder structure.")

        exit()

with open(
    "model/class_names.json",
    "w"
) as file:
    json.dump(class_name, 
              file indent = 4)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_szie = AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size = AUTOTUNE
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ]
)

model = keras.Sequential([

    layers.Input(Shape = (128, 128, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Convo2D(32, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(2, 2),
    layers.Convo2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation = 'softmax')

])

