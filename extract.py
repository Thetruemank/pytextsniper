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


def extract_text(image):
    if image is None or image.size == 0:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text


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

        # Draw the rectangle on the image copy if the coordinates are available
        if len(rect_coords) >= 4:
            thickness = 2
            color = (0, 0, 255)
            cv2.rectangle(image_copy, (rect_coords[0], rect_coords[1]), (rect_coords[2], rect_coords[3]), color, thickness)

        # Show the image with the rectangle
        cv2.imshow(window_name, image_copy)

        # Bring the window to the front
        window = gw.getWindowsWithTitle(window_name)[0]
        window.activate()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


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
