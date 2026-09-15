# Import Flask for creating the web application.
from flask import Flask, render_template, request

# Import TensorFlow to load the trained CNN model.
import tensorflow as tf

# Import NumPy to work with image pixel arrays.
import numpy as np

# Import PIL to open and process uploaded images.
from PIL import Image

# Import os to work with folders and file paths.
import os

# Import json to load the saved class names.
import json


# Create the Flask application.
app = Flask(__name__)


# Folder where uploaded images will be stored.
UPLOAD_FOLDER = "static/uploads"

# Create the folder if it does not already exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load the trained CNN model for prediction.
model = tf.keras.models.load_model(
    "model/leaf_disease_model.keras"
)


# Load the disease class names saved during training.
with open("model/class_names.json", "r") as file:
    class_names = json.load(file)


# Display the available classes.
print("Number of classes:", len(class_names))


# Create a preprocessing function for uploaded images.
def preprocess_image(image):

    # Convert the image to RGB format.
    image = image.convert("RGB")

    # Resize the image to the size used during training.
    image = image.resize((128, 128))

    # Convert the image into a NumPy array.
    image_array = np.array(image)

    # Convert pixel values to float32.
    image_array = image_array.astype("float32")

    # Add batch dimension: (128,128,3) → (1,128,128,3).
    return np.expand_dims(image_array, axis=0)


# Display the home page.
@app.route("/")
def home():

    # Load index.html.
    return render_template("index.html")


# Handle image upload and prediction.
@app.route("/predict", methods=["POST"])
def predict():

    # Get the uploaded image from the form.
    file = request.files.get("image")

    # Check whether an image was selected.
    if file is None or file.filename == "":
        return "Please select an image."

    # Create the path where the image will be saved.
    image_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    # Save the uploaded image.
    file.save(image_path)

    # Open the saved image using PIL.
    try:
        image = Image.open(image_path)

    except Exception:
        return "Invalid image file."

    # Prepare the image for the CNN.
    processed_image = preprocess_image(image)

    # Get prediction probabilities from the CNN.
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Find the class with the highest probability.
    predicted_index = np.argmax(predictions[0])

    # Get the disease name using the predicted index.
    predicted_class = class_names[predicted_index]

    # Convert the probability into a percentage.
    confidence = predictions[0][predicted_index] * 100

    # Make the dataset class name easier to read.
    display_name = predicted_class.replace(
        "___",
        " - "
    ).replace(
        "_",
        " "
    )

    # Send the prediction result to result.html.
    return render_template(
        "result.html",
        prediction=display_name,
        confidence=round(float(confidence), 2),
        image_path=image_path
    )


# Start Flask when this file is executed directly.
if __name__ == "__main__":

    # Run the Flask development server.
    app.run(debug=True)