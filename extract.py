from PIL import ImageGrab as ig
import cv2
import numpy as np
import pyperclip
import pytesseract
import keyboard
import threading
import pygetwindow as gw

# Set the path to the Tesseract executable file
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def capture_screen():
    screen = np.array(ig.grab(bbox=None))
    return cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)


def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to convert the image to black and white
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Dilate the image to join parts of the text that might have been separated due to the background
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Apply noise removal with morphological transformations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply edge detection
    edges = cv2.Canny(opening, 100, 200)

    return edges

def extract_text(image, lang='eng', ocr_config=''):
    if image is None or image.size == 0:
        return ""

    preprocessed = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed, lang=lang, config=ocr_config)
    return text


def draw_rounded_rectangle(image, rect_coords, color, thickness, radius, transparency):
    # Draw the outer rounded rectangle
    rect_img = np.zeros_like(image)
    cv2.rectangle(rect_img, (rect_coords[0] + radius, rect_coords[1] + radius), (rect_coords[2] - radius, rect_coords[3] - radius), color, -1)
    cv2.circle(rect_img, (rect_coords[0] + radius, rect_coords[1] + radius), radius, color, -1)
    cv2.circle(rect_img, (rect_coords[0] + radius, rect_coords[3] - radius), radius, color, -1)
    cv2.circle(rect_img, (rect_coords[2] - radius, rect_coords[1] + radius), radius, color, -1)
    cv2.circle(rect_img, (rect_coords[2] - radius, rect_coords[3] - radius), radius, color, -1)

    # Combine the original image with the rounded rectangle
    image = cv2.addWeighted(image, transparency, rect_img, 1 - transparency, 0)

    # Draw the outer border
    cv2.rectangle(image, (rect_coords[0] + radius, rect_coords[1]), (rect_coords[2] - radius, rect_coords[1] + thickness), color, -1)
    cv2.rectangle(image, (rect_coords[0] + radius, rect_coords[3] - thickness), (rect_coords[2] - radius, rect_coords[3]), color, -1)
    cv2.rectangle(image, (rect_coords[0], rect_coords[1] + radius), (rect_coords[0] + thickness, rect_coords[3] - radius), color, -1)
    cv2.rectangle(image, (rect_coords[2] - thickness, rect_coords[1] + radius), (rect_coords[2], rect_coords[3] - radius), color, -1)
    cv2.circle(image, (rect_coords[0] + radius, rect_coords[1] + radius), radius, color, -1)
    cv2.circle(image, (rect_coords[0] + radius, rect_coords[3] - radius), radius, color, -1)
    cv2.circle(image, (rect_coords[2] - radius, rect_coords[1] + radius), radius, color, -1)
    cv2.circle(image, (rect_coords[2] - radius, rect_coords[3] - radius), radius, color, -1)

    return image
        
def draw_rectangle(image):
    # Create a separate window to display the image and handle the mouse events
    window_name = "Image"
    cv2.namedWindow(window_name)

    # Initialize a variable to store the coordinates of the rectangle
    rect_coords = []
    rectangles = []

    # Define a function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal rect_coords

        # If the left mouse button is pressed, record the starting coordinates of the rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_coords[:] = [x, y, x, y]

        # If the left mouse button is moved, update the ending coordinates of the rectangle
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            rect_coords[2:] = [x, y]

        # If the left mouse button is released, record the ending coordinates of the rectangle
        elif event == cv2.EVENT_LBUTTONUP:
            rect_coords[2:] = [x, y]
            rectangles.append(tuple(rect_coords))

    # Add the mouse event listener to the window
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        # Create a copy of the original image to avoid modifying it
        image_copy = image.copy()

        # Draw the rounded rectangle on the image copy if the coordinates are available
        for rect in rectangles:
            color = (127, 127, 127)  # Grey color
            thickness = 2
            radius = 10  # Change this value to control the roundness of the rectangle's corners
            image_copy = draw_rounded_rectangle(image_copy, rect, color, thickness, radius, transparency=0.9)

        # Show the image with the rounded rectangle
        cv2.imshow(window_name, image_copy)

        # Try to bring the window to the front
        windows = gw.getWindowsWithTitle(window_name)
        if windows:
            window = windows[0]
            try:
                window.activate()
            except gw.PyGetWindowException as e:
                print("Error activating window:", e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    extracted_text = ""
    for rect in rectangles:
        cropped_image = image[rect[1]:rect[3], rect[0]:rect[2]]
        text = extract_text(cropped_image)
        if text and text.strip():
            extracted_text += text + "\n"

    if extracted_text.strip():
        # Display the extracted text in a separate window
        cv2.namedWindow("Extracted Text")
        cv2.imshow("Extracted Text", np.zeros((300, 600, 3), dtype=np.uint8))
        cv2.displayOverlay("Extracted Text", extracted_text, 0)
        cv2.waitKey(0)
        cv2.destroyWindow("Extracted Text")

        # Copy the extracted text to the clipboard
        pyperclip.copy(extracted_text)
        print("Text copied to clipboard:", extracted_text)

    return image


def main():
    # Wait for the Print Screen key to be pressed
    keyboard.wait('print_screen')

    # Capture the screen
    screen = capture_screen()

    # Draw a rectangle over the area with text and extract the text
    draw_rectangle(screen)


if __name__ == "__main__":
    while True:
        # Start a new thread to run the main function
        t = threading.Thread(target=main)
        t.start()

        # Wait for the thread to finish before continuing
        t.join()
