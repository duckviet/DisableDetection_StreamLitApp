import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("./best.pt")

# Streamlit title
st.title("Disable People Detection App")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Detection threshold
# thresh = 0.6
thresh = st.slider("Thresh hold", 0.0, 1.0, 0.5, 0.05)

# Function to run the detection and draw bounding boxes
def detect_and_draw(image, model, thresh):
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img_cv2)[0]

    if results.boxes.data.tolist():
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > thresh:
                label = model.names[int(class_id)].upper()
                color = (0, 255, 0) if class_id == 1.0 else (255, 0, 0)

                # Draw the bounding box and label
                cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img_cv2, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                return None
        # Convert back to PIL format for display
        img_output = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        return img_output
    else:
        return None

# Check if an image has been uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns for displaying images
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a button to initiate detection
    if st.button('Detect'):
        img_output = detect_and_draw(image, model, thresh)
        
        if img_output is not None:
            with col2:
                st.image(img_output, caption="Processed Image", use_column_width=True)
        else:
            st.warning("No objects detected.")
    else:
        st.info("Click 'Detect' to run object detection.")
