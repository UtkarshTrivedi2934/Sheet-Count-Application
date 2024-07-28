import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def count_sheets(image):
    # Convert the uploaded image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    # Count the number of contours
    sheet_count = len(contours)
    
    return sheet_count, contour_image

st.title("Sheet Counter")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = np.array(Image.open(uploaded_file))

    # Count the sheets and get the contour image
    sheet_count, contour_image = count_sheets(image)

    # Display the results
    st.write(f"Number of sheets: {sheet_count}")

    # Convert the contour image from BGR to RGB for displaying
    contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
    
    # Display the image with contours
    st.image(contour_image_rgb, caption="Annotated Image with Contours", use_column_width=True)
