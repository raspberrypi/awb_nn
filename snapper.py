
#! /usr/bin/env python3

import sys
import os
import argparse
import shutil
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QLineEdit, QPushButton, QLabel, QFileDialog,
                            QDialog, QDialogButtonBox, QScrollArea, QHBoxLayout,
                            QMessageBox)
from PyQt5.QtGui import QPixmap, QWheelEvent, QPainter, QPalette, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from picamera2 import Picamera2, Preview
from picamera2.previews.qt import QGlPicamera2, QPicamera2

# You can override these here, if you wish, or on the command line.
USER = ""
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "awb-captures")
CAMERA = 0

class Snapper(QMainWindow):
    def __init__(self, user=USER, output_dir=OUTPUT_DIR, camera=CAMERA, ssh_mode=False, initial_scene_id=0):
        super().__init__()

        self.output_dir = output_dir
        self.user = user
        self.scene_id = initial_scene_id  # Initialize scene ID counter with provided value

        self.configure_camera(camera)

        self.setWindowTitle("AWB Snapper")
        self.setGeometry(50, 50, 1000, 800)  # Increased main window size

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add EV control buttons
        self.ev_value = 0
        ev_button_layout = QHBoxLayout()

        self.capture_button = QPushButton("Capture")
        self.capture_button.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: 2px solid #388E3C;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                border-color: #9E9E9E;
            }
        """)
        self.capture_button.clicked.connect(self.capture)
        ev_button_layout.addWidget(self.capture_button)

        ev_down_button = QPushButton("EV-")
        ev_down_button.clicked.connect(self.ev_down)
        ev_button_layout.addWidget(ev_down_button)

        ev_up_button = QPushButton("EV+")
        ev_up_button.clicked.connect(self.ev_up)
        ev_button_layout.addWidget(ev_up_button)

        self.ev_value_label = QLabel(f"EV: {self.ev_value}")
        ev_button_layout.addWidget(self.ev_value_label)

        layout.addLayout(ev_button_layout)

        hbox_layout = QHBoxLayout()

        # Display USER value
        user_label = QLabel(f"User: {user}")
        hbox_layout.addWidget(user_label)

        # Display sensor information
        sensor_label = QLabel(f"Sensor: {self.sensor}")
        hbox_layout.addWidget(sensor_label)

        # Add scene ID display
        self.scene_id_label = QLabel(f"Scene Id: {self.scene_id:05d}")
        hbox_layout.addWidget(self.scene_id_label)

        # Display output directory at the bottom
        output_dir_label = QLabel(f"Output directory: {output_dir}")
        hbox_layout.addWidget(output_dir_label)

        layout.addLayout(hbox_layout)

        bg_colour = self.palette().color(QPalette.Background).getRgb()[:3]
        if ssh_mode:
            self.qpicamera2 = QPicamera2(self.picam2, bg_colour=bg_colour)
        else:
            self.qpicamera2 = QGlPicamera2(self.picam2, bg_colour=bg_colour)

        self.qpicamera2.done_signal.connect(self.capture_done)

        self.qpicamera2.setFixedSize(768, 512)
        layout.addWidget(self.qpicamera2)

        self.picam2.start()
        self.showMaximized()

    def configure_camera(self, camera):
        self.picam2 = Picamera2(camera)
        self.sensor = self.picam2.camera_properties['Model']
        if 'mono' in self.sensor.lower() or 'noir' in self.sensor.lower():
            raise ValueError("Mono/Noir cameras are not supported - please use a colour camera")
        self.capture_config = self.picam2.create_still_configuration()
        full_res = self.picam2.sensor_resolution
        half_res = (full_res[0] // 2, full_res[1] // 2)
        preview_res = half_res
        while preview_res[0] > 1280:
            preview_res = (preview_res[0] // 2, preview_res[1] // 2)
        self.preview_res = preview_res
        print(f"Preview resolution: {preview_res}")
        preview_config = self.picam2.create_preview_configuration(
            {'format': 'YUV420', 'size': preview_res},
            raw={'format': 'SBGGR12', 'size': half_res}, # force unpacked, full FOV
            controls={'FrameRate': 30}
        )
        self.picam2.configure(preview_config)
        if 'AfMode' in self.picam2.camera_controls:
            self.picam2.set_controls({"AfMode": 2})  # Continuous AF, where available

    def ev_up(self):
        self.ev_value += 0.125
        self.picam2.set_controls({"ExposureValue": self.ev_value})
        self.ev_value_label.setText(f"EV: {self.ev_value}")

    def ev_down(self):
        self.ev_value -= 0.125
        self.picam2.set_controls({"ExposureValue": self.ev_value})
        self.ev_value_label.setText(f"EV: {self.ev_value}")

    def capture(self):
        self.capture_button.setEnabled(False)
        print("Doing capture")
        self.picam2.switch_mode_and_capture_request(
            self.capture_config, wait=False, signal_function=self.qpicamera2.signal_done)

    def capture_done(self, job):
        self.capture_button.setEnabled(True)
        request = job.get_result()
        while True:
            filename = os.path.join(self.output_dir, f"{self.user},{self.sensor},{self.scene_id:05d}")
            if not os.path.exists(filename + ".jpg"):
                break
            self.scene_id += 1
            self.scene_id_label.setText(f"Scene Id: {self.scene_id:05d}")
        request.save('main', filename + ".jpg")
        request.save_dng(filename + ".dng")
        print("Capture done", request)
        request.release()
        print("Files saved as", filename + ".jpg and", filename + ".dng")

        # Increment scene ID and update display
        self.scene_id += 1
        self.scene_id_label.setText(f"Scene Id: {self.scene_id:05d}")

    def is_valid_filename(self, text):
        # List of characters not allowed in filenames
        invalid_chars = '<>:"/\\|?*,\''
        return not any(char in invalid_chars for char in text)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AWB-O-Matic Tool')
    parser.add_argument('-u', '--user', help='Set the user name for saved images')
    parser.add_argument('-o', '--output', help='Override the output directory')
    parser.add_argument('--initial-scene-id', type=int, default=0, help='Initial scene ID value (default: 0)')
    ssh_group = parser.add_mutually_exclusive_group()
    ssh_group.add_argument('-s', '--ssh', action='store_true', help='Enable SSH mode')
    ssh_group.add_argument('--no-ssh', action='store_true', help='Disable SSH mode')
    args = parser.parse_args()

    # Override USER if command line argument is provided
    if args.user:
        USER = args.user

    # Override OUTPUT_DIR if command line argument is provided
    if args.output:
        OUTPUT_DIR = args.output

    # Check if USER is set
    if not USER:
        parser.error("User name must be set. Use -u/--user to specify a user name.")

    # Check for invalid characters in USER
    invalid_chars = '<>:"/\\|?*,\''
    if any(char in invalid_chars for char in USER):
        parser.error(f"User name contains invalid characters. Please avoid: {invalid_chars}")

    # Set SSH mode based on arguments or environment
    ssh_mode = None
    if args.ssh:
        ssh_mode = True
    elif args.no_ssh:
        ssh_mode = False
    else:
        # Try to deduce SSH status from DISPLAY environment variable
        display = os.environ.get('DISPLAY', '')
        if display.startswith('localhost:') or display.startswith('127.0.0.1:'):
            ssh_mode = True
        elif display:
            ssh_mode = False

    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"User: {USER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"SSH mode: {ssh_mode}")

    app = QApplication(sys.argv)
    window = Snapper(user=USER, output_dir=OUTPUT_DIR, ssh_mode=ssh_mode, initial_scene_id=args.initial_scene_id)
    window.show()
    sys.exit(app.exec_())
