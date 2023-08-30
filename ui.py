from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QProgressBar, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from extract import capture_screen, extract_text

class TextExtractionThread(QThread):
    text_extracted = pyqtSignal(str)

    def run(self):
        screen = capture_screen()
        text = extract_text(screen)
        self.text_extracted.emit(text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel()
        self.capture_button = QPushButton("Capture Screen")
        self.progress_bar = QProgressBar()

        self.capture_button.clicked.connect(self.capture_screen)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def capture_screen(self):
        self.capture_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.thread = TextExtractionThread()
        self.thread.text_extracted.connect(self.display_text)
        self.thread.start()

    def display_text(self, text):
        self.image_label.setText(text)
        self.progress_bar.setValue(100)
        self.capture_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
