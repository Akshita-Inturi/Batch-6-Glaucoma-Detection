import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model("C:/Users/akshi/Music/Detection/vgg19_model.h5")  # Replace with the actual path to your model file

# Define the function to make predictions
def predict_image(img):
    # Convert Gradio Image datatype to PIL Image
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    # Preprocess the image
    img = img.resize((128, 128))  # Resize image to match model's input size
    img = np.array(img)  # Convert image to numpy array
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    output = model.predict(img)

    # Interpret prediction
    if output[0][0] > output[0][1]:
        pred = "Glaucoma Negative"
    else:
        pred = "Glaucoma Positive"

    return pred

# Create a Gradio interface
interface = gr.Interface(fn=predict_image, inputs="image", outputs="label", title="Glaucoma Classifier")

# Launch the interface
interface.launch()