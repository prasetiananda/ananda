import cv2
import streamlit as st
import numpy as np

def detect_faces(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haarcascades face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

def main():
    st.title("Face Detection App")

    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the selected image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform face detection
        result_image = detect_faces(image)

        # Display the result image with detected faces
        st.image(result_image, caption="Detected Faces", use_column_width=True)

if __name__ == "__main__":
    main()
