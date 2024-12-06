import streamlit as st
import os
from inference import main
from PIL import Image
import numpy as np
import subprocess

# Directory for uploaded images
IMAGE_DIR = './ip/'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


# Check and execute model conversion
def check_and_convert_model():
    model_path = "./wav2lip_openvino_model.xml"
    if not os.path.exists(model_path):
        st.warning("OpenVINO model not found. Converting model...")
        try:
            result = subprocess.run(["python", "model_convert.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Model conversion successful!")
            else:
                st.error(f"Model conversion failed. Error: {result.stderr}")
        except Exception as e:
            st.error(f"Error during model conversion: {e}")

# Function to clear directory
def clear_directory(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Streamlit App
def app():
    st.title("Real-Time Lip-Sync Application")

    # Sidebar controls
    st.sidebar.title("Controls")
    start_inference = st.sidebar.button("Start Lip-Sync Inference")
    stop_inference = st.sidebar.button("Stop Inference")
    clear_images = st.sidebar.button("Clear Uploaded Images")

    # Upload Image
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    file_path = None

    if uploaded_file is not None:
        # Clear previously uploaded images
        clear_directory(IMAGE_DIR)

        # Save and display the uploaded file
        file_path = os.path.join(IMAGE_DIR, "test_1.jpg")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = Image.open(file_path)
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated to use_container_width
        st.success(f"Image {uploaded_file.name} uploaded successfully!")

    # Placeholder for the video stream
    st.header("Real-Time Video Stream")
    video_placeholder = st.empty()

    # Control logic
    global inference_flag
    flag = 0

    # Check and convert model before starting inference
    check_and_convert_model()
    if start_inference:
        if file_path:
            st.write("Starting inference...")
            flag = 1
            try:
                st.write("Inference started.")
                # Call main with the image file path and flag
                for frame in main(file_path, flag):
                    # Debugging the frame shape and type
                    #st.write(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    if isinstance(frame, np.ndarray):
                        if frame.ndim == 3 and frame.shape[2] == 3:
                            #st.write("Frame is in RGB format.")
                            video_placeholder.image(frame, channels="RGB", use_container_width=True)  # Updated to use_container_width
                        else:
                            st.error(f"Frame is not in RGB format. Shape: {frame.shape}")
                    else:
                        st.error(f"Unexpected frame type: {type(frame)}")

                    # Check if the flag was set to stop
                    if flag == 0:
                        st.info("Inference stopped.")
                        break
            except Exception as e:
                st.error(f"Inference error: {e}")
        else:
            st.warning("Please upload an image before starting inference.")

    if stop_inference:
        flag = 0
        st.info("Inference stopped.")

    if clear_images:
        clear_directory(IMAGE_DIR)
        st.success("Cleared uploaded images.")

if __name__ == "__main__":
    app()
