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


from PyQt5.QtGui import QImage

def capture_screen():
    screen = np.array(ig.grab(bbox=None))
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    height, width, channel = screen.shape
    bytesPerLine = 3 * width
    return QImage(screen.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()


def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to convert the image to black and white
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Dilate the image to join parts of the text that might have been separated due to the background
    dilated = cv2.dilate(thresh, None, iterations=2)

    return dilated

from PyQt5.QtCore import QThread, pyqtSignal

class TextExtractor(QThread):
    text_extracted = pyqtSignal(str)

    def __init__(self, image):
        super().__init__()
        self.image = image

    def run(self):
        if self.image is None or self.image.size == 0:
            self.text_extracted.emit("")
            return

        preprocessed = preprocess_image(self.image)
        text = pytesseract.image_to_string(preprocessed)
        self.text_extracted.emit(text)


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




from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    while True:
        # Start a new thread to run the main function
        t = threading.Thread(target=main)
        t.start()

        # Wait for the thread to finish before continuing
        t.join()
