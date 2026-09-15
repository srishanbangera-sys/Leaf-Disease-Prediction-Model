import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json

DATASET_PATH = "dataset/Plantvillage"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10


if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset was not found.")
    exit()


train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True,
    batch_size=BATCH_SIZE
)


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=False,
    batch_size=BATCH_SIZE
)


# Get class names
class_names = train_dataset.class_names

# Print classes
for index, name in enumerate(class_names):
    print(index, ":", name)

# Get number of classes
num_classes = len(class_names)

print("Number of classes:", num_classes)


if num_classes < 2:
    print("Error: Only one class was detected.")
    print("Check your dataset folder structure.")
    exit()


# Save class names
with open("model/class_names.json", "w") as file:
    json.dump(
        class_names,
        file,
        indent=4
    )


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ]
)


# Model
model = keras.Sequential([

    layers.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        num_classes,
        activation="softmax"
    )

])


# Compile
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# Save model
model.save(
    "model/plant_disease_model.keras"
)

print("Model saved Successfully")