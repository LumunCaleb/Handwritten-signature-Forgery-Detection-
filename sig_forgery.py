import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load the YOLO model (adjust the path to your model)
model = YOLO(r"C:\Users\Caleb\Downloads\Signature_forgery_model.pt")  # Replace with your model path

# Streamlit app title
st.title("Signature Forgery Detection")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload a signature image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image as a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name, format="JPEG")
        temp_image_path = tmp_file.name

    # Perform prediction
    st.write("Running prediction...")
    results = model.predict(source=temp_image_path)  # Pass the temp file path to YOLO

    # Display prediction results
    for result in results:
        predicted_class = result.names[result.probs.top1]  # Top-1 class name
        confidence_score = result.probs.top1conf            # Confidence score

        # Display the results
        st.write(f"RESULT: {predicted_class} signature")
        st.write(f"Confidence: {confidence_score:.2f}")

    # Clean up temporary file
    os.remove(temp_image_path)

else:
    st.write("Please upload an image to start the prediction.")

