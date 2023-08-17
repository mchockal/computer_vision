import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Create application title and file uploader widget.
st.title("Face Detection and Blurring")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])


# Function for detecting facses in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections


# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, factor=3, blur_face=False, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])

            if blur_face:
                face = frame[y1:y2, x1:x2]
                face = blur(face, scale_factor=factor)
                frame[y1:y2, x1:x2] = face
            # Scale bounding box thickness based on frame height
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes


# Function to load the DNN model.
@st.cache_resource()
def load_model():
    modelFile = "resources/face_recognition/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "resources/face_recognition/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Function to blur the detected faces in image
def blur(face, scale_factor=3):

    h = face.shape[0]
    w = face.shape[1]

    w_k = int(w/scale_factor)
    h_k = int(h/scale_factor)

    if w_k % 2 == 0:
        w_k += 1
    if h_k % 2 == 0:
        h_k += 1

    blurred_face = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
    return blurred_face

net = load_model()

if img_file_buffer is not None:
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # Or use PIL Image (which uses an RGB channel order)
    # image = np.array(Image.open(img_file_buffer))

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    blur_face = st.checkbox("Blur Face(s)")

    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Create a Slider and get the threshold from the slider.
    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # Call the face detection model to detect faces in the image.
    detections = detectFaceOpenCVDnn(net, image)

    scale_factor = 0
    if blur_face:
        scale_factor = st.slider("Set blur scale factor", min_value=1, max_value=5, step=1, value=3)
    # Process the detections based on the current confidence threshold.
    out_image, _ = process_detections(image, detections, factor=scale_factor, blur_face=blur_face, conf_threshold=conf_threshold)

    # Display Detected faces.
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    # Convert opencv image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'),
                unsafe_allow_html=True)
