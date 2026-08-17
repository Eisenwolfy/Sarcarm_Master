import os
import sys

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton,
    QProgressBar, QApplication,
)
from PyQt5.QtGui import QMovie, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from models import lstm_bert_cnn.py

BACKGROUND_GIF = "sad-koi.gif"

class ModelLoaderThread(QThread):
    finished_ok = pyqtSignal()
    finished_error = pyqtSignal(str)

    def run(self):
        try:
            model.load_artifacts()
            self.finished_ok.emit()
        except Exception as e:
            self.finished_error.emit(str(e))


class PredictThread(QThread):
    finished_ok = pyqtSignal(str, float)
    finished_error = pyqtSignal(str)

    def __init__(self, comment, parent_comment):
        super().__init__()
        self.comment = comment
        self.parent_comment = parent_comment

    def run(self):
        try:
            label, prob = model.predict(self.comment, self.parent_comment)
            self.finished_ok.emit(label, prob)
        except Exception as e:
            self.finished_error.emit(str(e))


class GifBackgroundApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_ready = False
        self.initUI()
        self._load_model_async()

    def initUI(self):
        width, height = 889, 500
        self.setFixedSize(width, height)
        self.setWindowTitle("Sarcasm Detector")
        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, width, height)

        if os.path.exists(BACKGROUND_GIF):
            self.movie = QMovie(BACKGROUND_GIF)
            self.bg_label.setMovie(self.movie)
            self.bg_label.setScaledContents(True)
            self.movie.start()
        else:
            self.setStyleSheet("background-color: #1a1a1a;")

        self.title_label = QLabel("Sarcasm Detector")
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Loading model…")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.input_field = QLineEdit()
        self.input_field.setFixedSize(width - 80, 50)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        self.input_field.setPlaceholderText("Paste a comment to check…")
        self.input_field.returnPressed.connect(self.handle_submit)

        self.parent_field = QLineEdit()
        self.parent_field.setFixedSize(width - 80, 40)
        self.parent_field.setStyleSheet("""
            QLineEdit {
                background-color: #232323;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 5px;
                font-size: 13px;
            }
        """)
        self.parent_field.setPlaceholderText("Optional — what they were replying to")
        self.parent_field.returnPressed.connect(self.handle_submit)

        self.button = QPushButton("Check")
        self.button.setFixedSize(150, 50)
        self.button.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.button.setEnabled(False)
        self.button.clicked.connect(self.handle_submit)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: white;
                border-radius: 4px;
                font-weight: bold;
                border: 2px solid #555555;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QPushButton:pressed:enabled {
                background-color: #367c39;
            }
            QPushButton:disabled {
                background-color: #1f1f1f;
                color: #777777;
            }
        """)

        self.result_label = QLabel("Result will appear here")
        self.result_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setFixedSize(width - 80, 18)
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 5px;
                color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4ade80;
                border-radius: 5px;
            }
        """)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addSpacing(20)
        layout.addWidget(self.input_field, alignment=Qt.AlignCenter)
        layout.addSpacing(8)
        layout.addWidget(self.parent_field, alignment=Qt.AlignCenter)
        layout.addWidget(self.button, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        layout.addWidget(self.result_label)
        layout.addWidget(self.confidence_bar, alignment=Qt.AlignCenter)
        layout.addStretch(1)

        self.setLayout(layout)

    def _load_model_async(self):
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.finished_ok.connect(self._on_model_loaded)
        self.loader_thread.finished_error.connect(self._on_model_load_error)
        self.loader_thread.start()

    def _on_model_loaded(self):
        self.model_ready = True
        self.status_label.setText("Model ready.")
        self.button.setEnabled(True)

    def _on_model_load_error(self, message):
        self.status_label.setText("Failed to load model.")
        self.result_label.setText(f"Error: {message}")
        self.result_label.setStyleSheet("color: #f87171; font-size: 13px; font-weight: bold;")

    def handle_submit(self):
        if not self.model_ready:
            return

        user_text = self.input_field.text().strip()
        parent_text = self.parent_field.text().strip()
        if not user_text:
            self.result_label.setText("You didn't type anything!")
            self.result_label.setStyleSheet("color: #f87171; font-size: 16px; font-weight: bold;")
            return

        self.button.setEnabled(False)
        self.button.setText("…")
        self.result_label.setText("Thinking…")
        self.result_label.setStyleSheet("color: #aaaaaa; font-size: 16px; font-weight: bold;")
        self.confidence_bar.setValue(0)

        self.predict_thread = PredictThread(user_text, parent_text)
        self.predict_thread.finished_ok.connect(self._on_prediction_done)
        self.predict_thread.finished_error.connect(self._on_prediction_error)
        self.predict_thread.start()

    def _on_prediction_done(self, label, probability):
        self.button.setEnabled(True)
        self.button.setText("Check")

        pct = round(probability * 100)
        is_sarcasm = label == "sarcasm"

        self.result_label.setText("Sarcasm detected" if is_sarcasm else "Not sarcasm")
        self.result_label.setStyleSheet(
            f"color: {'#f87171' if is_sarcasm else '#4ade80'}; font-size: 16px; font-weight: bold;"
        )

        self.confidence_bar.setValue(pct)
        self.confidence_bar.setFormat(f"{pct}% confidence")
        chunk_color = "#f87171" if is_sarcasm else "#4ade80"
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 5px;
                color: white;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
                border-radius: 5px;
            }}
        """)

    def _on_prediction_error(self, message):
        self.button.setEnabled(True)
        self.button.setText("Check")
        self.result_label.setText(f"Error: {message}")
        self.result_label.setStyleSheet("color: #f87171; font-size: 13px; font-weight: bold;")


def main():
    app = QApplication(sys.argv)
    ex = GifBackgroundApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
