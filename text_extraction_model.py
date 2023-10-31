import cv2
import re
import sys
import numpy as np
import os
from google.cloud import vision_v1
from google.cloud.vision_v1 import types

def preprocess_image(image):
    # image = cv2.imread(image_path)
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)


    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    g_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply thresholding for binarization
    block_size = 5
    C = 2
    thresh_image = cv2.adaptiveThreshold(g_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)

    return thresh_image


def post_process_text(extracted_text):
    cleaned_text = extracted_text 
    cleaned_text = ' '.join(extracted_text.split())
    cleaned_text = re.sub(r'\b1\b', 'l', cleaned_text)
    cleaned_text = re.sub(r'\b0\b', 'O', cleaned_text)

    # Correcting bracket spacing
    cleaned_text = re.sub(r'(\w)\(', r'\1 (', cleaned_text)  # Add space after opening parenthesis
    cleaned_text = re.sub(r'\)\s+', ')', cleaned_text)  # Remove spaces after closing parenthesis
    cleaned_text = re.sub(r'(\w)\{', r'\1 {', cleaned_text)  # Add space after opening curly brace
    cleaned_text = re.sub(r'\}\s+', '}', cleaned_text)  # Remove spaces after closing curly brace
    cleaned_text = re.sub(r'\[\s+', '[', cleaned_text)  # Remove spaces after opening square bracket
    cleaned_text = re.sub(r'\s+\]', ']', cleaned_text)  # Remove spaces before closing square bracket
    cleaned_text = re.sub(r';\s+', '; ', cleaned_text)  # Add a space after semicolons

    # Correcting inconsistent casing
    cleaned_text = re.sub(r'\bif|If\b', 'if', cleaned_text)  # Ensure consistent casing for "if" statements
    cleaned_text = re.sub(r'\belse|Else\b', 'else', cleaned_text)  # Ensure consistent casing for "else" statements
    # Formatting code rules
    cleaned_text = re.sub(r'(\bif\b|\bfor\b)\s*{\s*', r'\1 {', cleaned_text)  # Ensure space after "if" or "for" followed by an opening brace

    return cleaned_text




def extract_text(image):
    
    # Preprocessing the image
    thresh_image = preprocess_image(image)

    # Connecting with the credentials JSON file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'text-detection-app-403206-e9ec799cc72e.json'
    client = vision_v1.ImageAnnotatorClient()

    # Converting the thresholded image to bytes
    image_bytes = cv2.imencode(".png", thresh_image)[1].tobytes()
    test_image = types.Image(content=image_bytes)

    # Performing text extraction
    extract_text_from_vision = client.text_detection(image=test_image)

    final_text = ""
    first_annotation = extract_text_from_vision.text_annotations[0] if extract_text_from_vision.text_annotations else None

    for txt in extract_text_from_vision.text_annotations:
        if first_annotation:
          final_text = first_annotation.description
        else:
          final_text = ""

    # sending for post processing of extracted text
    final_text = post_process_text(final_text)
    final_text = format_text(final_text)
    return final_text


def format_text(text):
    formatted_text = text
    formatted_text = re.sub(r'(\{|}|;)', r'\1\n', formatted_text)
    # Adding indentation 
    formatted_text = re.sub(r'\n', r'\n    ', formatted_text)

    return formatted_text

