#! /usr/bin/env python3

import sys
import os
import argparse
import shutil
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QLineEdit, QPushButton, QLabel, QFileDialog,
                            QDialog, QDialogButtonBox, QScrollArea, QHBoxLayout,
                            QMessageBox, QListWidget, QSplitter, QSizePolicy, QListWidgetItem, QSlider)
from PyQt5.QtGui import QPixmap, QWheelEvent, QPainter, QPalette, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QRect

from dng import Dng
from image_dialog import ImageDialog

# You can override these here, if you wish, or on the command line.
INPUT = os.path.join(os.path.expanduser("~"), "awb-images")

class Annotator(QMainWindow):
    def __init__(self, input=INPUT):
        super().__init__()
        self.setWindowTitle("AWB Annotator")
        self.setGeometry(100, 100, 600, 900)

        self.input = input

        # Track processed files
        self.processed_files = set()

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)  # Changed to QVBoxLayout

        # Add directory labels at the top
        dir_layout = QHBoxLayout()
        dir_layout.setContentsMargins(5, 2, 5, 2)  # Minimize vertical margins
        input_label = QLabel(f"Input: {self.input}")
        input_label.setStyleSheet("color: #666666; font-weight: bold;")
        # Set size policies to prevent vertical expansion
        input_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        dir_layout.addWidget(input_label)
        dir_layout.addStretch()
        self.main_layout.addLayout(dir_layout)

        # Add instruction label
        instruction_label = QLabel("Double-click a file to annotate it")
        instruction_label.setStyleSheet("color: #666666; font-style: italic;")
        instruction_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        instruction_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.main_layout.addWidget(instruction_label)

        # Create splitter for resizable panes
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Create file list widget
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(200)
        self.file_list.itemDoubleClicked.connect(self.on_file_double_clicked)
        # Set a brighter background color
        self.file_list.setStyleSheet("background-color: #3D3D3D; color: #FFFFFF;")
        self.splitter.addWidget(self.file_list)

        # Create main content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.splitter.addWidget(self.content_area)

        # Load files
        self.load_files()

    def load_files(self):
        """Load DNG files from input directory into the list widget"""
        self.file_list.clear()
        try:
            files = []

            # Check if input is a single file
            if os.path.isfile(self.input):
                if self.input.lower().endswith('.dng'):
                    files.append(self.input_dir)
            else:
                # Input is a directory, walk through all subdirectories
                for root, dirs, filenames in os.walk(self.input):
                    for filename in filenames:
                        if filename.lower().endswith('.dng'):
                            # Get the full path relative to input directory
                            full_path = os.path.join(root, filename)
                            relative_path = os.path.relpath(full_path, self.input)
                            files.append(relative_path)

            files.sort()
            num_files = 0
            for file in files:
                # Remove any existing checkmark from the filename
                base_name = file.split('/')[-1]
                display_name = file.replace("✓ ", "")
                item = QListWidgetItem(display_name)
                # Add checkmark if file has been processed
                unannotated_filename = f"{display_name.split(',')[0]},{display_name.split(',')[1]},{display_name.split(',')[2]}.dng"
                if unannotated_filename in self.processed_files:
                    item.setText(f"✓ {display_name}")
                self.file_list.addItem(item)
                num_files += 1
            print(num_files, "files loaded")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load files: {str(e)}")

    def on_file_double_clicked(self, item):
        """Handle double-click on a file in the list"""
        filename = item.text()
        self.process_file(filename)

    def process_file(self, filename):
        """Process the selected file by loading and displaying it in ImageDialog"""
        try:
            # Remove any checkmark from the filename
            clean_filename = filename.replace("✓ ", "")
            base_name = clean_filename[:-4]  # Remove .dng extension (which we know it to have)
            parts = base_name.split(',')
            unannotated_filename = f"{parts[0]},{parts[1]},{parts[2]}.dng"

            # Construct full path to the image
            image_path = os.path.join(self.input, clean_filename)

            dng = Dng(image_path, sensor=parts[1])
            print("Camera white balance:", dng.camera_white_balance)
            colour_gains = np.array(dng.camera_white_balance)[[0, 2]]

            def is_non_negative_integer(s):
                return s.isdigit() and int(s) >= 0

            def is_positive_number(s):
                try:
                    return float(s) > 0
                except ValueError:
                    return False

            # Check if filename has already been annotated with colour gains
            if len(parts) == 5 and all(is_positive_number(part) for part in parts[-2:]):
                colour_gains = [float(part) for part in parts[-2:]]
                print("Using annotated colour gains:", colour_gains)
            else:
                print("Using camera white balance gains:", colour_gains)

            # Create and show the image dialog
            dialog = ImageDialog(self, dng=dng, colour_gains=colour_gains)
            result = dialog.exec_()

            # After dialog is closed, check if it was accepted
            if result == QDialog.Accepted:
                # Add file to processed set
                self.processed_files.add(unannotated_filename)

                red_gain = dialog.colour_gains[0]
                blue_gain = dialog.colour_gains[1]
                print(f"Annotating file with gains: red {red_gain} blue {blue_gain}")

                new_filename = unannotated_filename.replace(".dng", f",{red_gain},{blue_gain}.dng")
                new_image_path = os.path.join(self.input, new_filename)

                # Rename the original file to the new name
                os.rename(image_path, new_image_path)
                print(f"Renamed {image_path} to {new_image_path}")

                self.load_files()
            else:
                print(f"Annotation cancelled for {clean_filename}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error processing file {clean_filename}: {str(e)}")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AWB Annotator')
    parser.add_argument('-i', '--input', type=str, default=INPUT,
                      help=f'Input directory containing images (default: {INPUT})')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = Annotator(input=args.input)
    window.show()
    sys.exit(app.exec_())