from PIL import ImageGrab as ig
import cv2
import numpy as np
import pyperclip
import pytesseract
import keyboard
import threading
import pygetwindow as gw
from deskew import skew_correction

# Set the path to the Tesseract executable file
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def capture_screen():
    screen = np.array(ig.grab(bbox=None))
    return cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)


def improve_image_quality(image):
    # Remove noise from the image
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Correct the skew of the image
    corrected = skew_correction(denoised)

    # Perform morphological operations
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(corrected, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply bilateral filter for smoothing
    smooth = cv2.bilateralFilter(blurred, 9, 75, 75)

    # Apply adaptive thresholding to convert the image to black and white
    thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Dilate the image to join parts of the text that might have been separated due to the background
    dilated = cv2.dilate(thresh, None, iterations=2)

    return dilated

def extract_text(image):
    if image is None or image.size == 0:
        return ""

    improved = improve_image_quality(image)
    preprocessed = preprocess_image(improved)
    data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
    
    # Filter out the text with low confidence levels and short length
    text = " ".join([data['text'][i] for i in range(len(data['conf'])) if int(data['conf'][i]) > 60 and len(data['text'][i]) > 2])
    
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
            cropped_image = image[rect_coords[1]                                  :rect_coords[3], rect_coords[0]:rect_coords[2]]
            text = extract_text(cropped_image)

        if text and text.strip():  # Check if the extracted text is not empty
            pyperclip.copy(text)
            print("Text copied to clipboard:", text)

    # Add the mouse event listener to the window
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        # Create a copy of the original image to avoid modifying it
        image_copy = image.copy()

        # Draw the rounded rectangle on the image copy if the coordinates are available
        if len(rect_coords) >= 4:
            color = (127, 127, 127)  # Grey color
            thickness = 2
            radius = 10  # Change this value to control the roundness of the rectangle's corners
            image_copy = draw_rounded_rectangle(image_copy, rect_coords, color, thickness, radius, transparency=0.9)

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

    return image
        
        def draw_rectangle(image):
        # Create a separate window to display the image and handle the mouse events
        window_name = "Image"
        cv2.namedWindow(window_name)

    # Draw the outer rounded rectangle
    rect_img = np.zeros_like(image)
    cv2.rectangle(rect_img, (x1 + radius, y1 + radius), (x2 - radius, y2 - radius), color, -1)
    cv2.circle(rect_img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(rect_img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(rect_img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(rect_img, (x2 - radius, y2 - radius), radius, color, -1)

    # Combine the original image with the rounded rectangle
    image = cv2.addWeighted(image, transparency, rect_img, 1 - transparency, 0)

    # Draw the outer border
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y1 + thickness), color, -1)
    cv2.rectangle(image, (x1 + radius, y2 - thickness), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x1 + thickness, y2 - radius), color, -1)
    cv2.rectangle(image, (x2 - thickness, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)




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
