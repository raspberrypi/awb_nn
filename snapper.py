
#! /usr/bin/env python3

import sys
import os
import argparse
import shutil
import struct
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QLineEdit, QPushButton, QLabel, QFileDialog,
                            QDialog, QDialogButtonBox, QScrollArea, QHBoxLayout,
                            QMessageBox)
from PyQt5.QtGui import QPixmap, QWheelEvent, QPainter, QPalette, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from picamera2 import Picamera2, Preview
from picamera2.previews.qt import QGlPicamera2, QPicamera2
from pidng.camdefs import Picamera2Camera
from pidng.core import PICAM2DNG
from pathlib import Path

# You can override these here, if you wish, or on the command line.
USER = ""
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "awb-captures")
CAMERA = 0

# PISP Constants
PISP_AWB_STATS_SIZE = 32
PISP_AWB_STATS_NUM_ZONES = PISP_AWB_STATS_SIZE * PISP_AWB_STATS_SIZE  # 1024
PISP_CDAF_STATS_SIZE = 8
PISP_CDAF_STATS_NUM_FOMS = PISP_CDAF_STATS_SIZE * PISP_CDAF_STATS_SIZE  # 64
PISP_FLOATING_STATS_NUM_ZONES = 4
PISP_AGC_STATS_NUM_BINS = 1024
PISP_AGC_STATS_SIZE = 16
PISP_AGC_STATS_NUM_ZONES = PISP_AGC_STATS_SIZE * PISP_AGC_STATS_SIZE  # 256
PISP_AGC_STATS_NUM_ROW_SUMS = 512

class Snapper(QMainWindow):
    def __init__(self, user=USER, output_dir=OUTPUT_DIR, camera=CAMERA, ssh_mode=False, initial_scene_id=0, small_output=None):
        super().__init__()

        self.output_dir = output_dir
        self.user = user
        self.scene_id = initial_scene_id  # Initialize scene ID counter with provided value
        self.small_output = small_output
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
            controls={'FrameRate': 30, 'StatsOutputEnable': 1}
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
        if self.small_output:
            small_filename = os.path.join(self.small_output, f"{self.user},{self.sensor},{self.scene_id:05d}")
            save_stats_dng(request, small_filename + ".dng")
        print("Capture done", request)
        request.release()
        print("Files saved as", filename + ".jpg and", filename + ".dng")
        if self.small_output:
            print("Statistics file saved as", small_filename + ".dng")

        # Increment scene ID and update display
        self.scene_id += 1
        self.scene_id_label.setText(f"Scene Id: {self.scene_id:05d}")

    def is_valid_filename(self, text):
        # List of characters not allowed in filenames
        invalid_chars = '<>:"/\\|?*,\''
        return not any(char in invalid_chars for char in text)

def decode_stats(stats):
    """
    Decode PISP statistics from binary data.

    Args:
        stats: Binary data containing PISP statistics

    Returns:
        dict: Decoded statistics with AWB, AGC, and CDAF data
    """
    # Convert to bytes if needed
    if not isinstance(stats, bytes):
        stats = bytes(stats)

    offset = 0
    result = {}

    # Decode AWB statistics
    awb_stats = {}

    # AWB zones (32x32 = 1024 zones)
    awb_zones = []
    for i in range(PISP_AWB_STATS_NUM_ZONES):
        if offset + 16 > len(stats):
            raise Exception("Stats too short")
        # pisp_awb_statistics_zone: R_sum, G_sum, B_sum, counted (4 x uint32)
        r_sum, g_sum, b_sum, counted = struct.unpack_from("<IIII", stats, offset)
        awb_zones.append({
            "R_sum": r_sum,
            "G_sum": g_sum,
            "B_sum": b_sum,
            "counted": counted
        })
        offset += 16

    # AWB floating zones
    awb_floating = []
    for i in range(PISP_FLOATING_STATS_NUM_ZONES):
        if offset + 16 > len(stats):
            raise Exception("Stats too short")
        r_sum, g_sum, b_sum, counted = struct.unpack_from("<IIII", stats, offset)
        awb_floating.append({
            "R_sum": r_sum,
            "G_sum": g_sum,
            "B_sum": b_sum,
            "counted": counted
        })
        offset += 16

    awb_stats["zones"] = awb_zones
    awb_stats["floating"] = awb_floating
    result["awb"] = awb_stats

    # Decode AGC statistics
    agc_stats = {}

    # AGC row sums (uint32 array)
    row_sums = []
    for i in range(PISP_AGC_STATS_NUM_ROW_SUMS):
        if offset + 4 > len(stats):
            raise Exception("Stats too short")
        row_sum = struct.unpack_from("<I", stats, offset)[0]
        row_sums.append(row_sum)
        offset += 4

    # AGC histogram (uint32 array)
    histogram = []
    for i in range(PISP_AGC_STATS_NUM_BINS):
        if offset + 4 > len(stats):
            raise Exception("Stats too short")
        bin_value = struct.unpack_from("<I", stats, offset)[0]
        histogram.append(bin_value)
        offset += 4

    # AGC floating zones
    agc_floating = []
    for i in range(PISP_FLOATING_STATS_NUM_ZONES):
        if offset + 16 > len(stats):
            raise Exception("Stats too short")
        # pisp_agc_statistics_zone: Y_sum (uint64), counted (uint32), pad (uint32)
        y_sum = struct.unpack_from("<Q", stats, offset)[0]
        offset += 8
        counted, pad = struct.unpack_from("<II", stats, offset)
        offset += 8
        agc_floating.append({
            "Y_sum": y_sum,
            "counted": counted,
            "pad": pad
        })

    agc_stats["row_sums"] = row_sums
    agc_stats["histogram"] = histogram
    agc_stats["floating"] = agc_floating
    result["agc"] = agc_stats

    # Decode CDAF statistics
    cdaf_stats = {}

    # CDAF foms (uint64 array)
    foms = []
    for i in range(PISP_CDAF_STATS_NUM_FOMS):
        if offset + 8 > len(stats):
            raise Exception("Stats too short")
        fom = struct.unpack_from("<Q", stats, offset)[0]
        foms.append(fom)
        offset += 8

    # CDAF floating (uint64 array)
    cdaf_floating = []
    for i in range(PISP_FLOATING_STATS_NUM_ZONES):
        if offset + 8 > len(stats):
            raise Exception("Stats too short")
        floating_val = struct.unpack_from("<Q", stats, offset)[0]
        cdaf_floating.append(floating_val)
        offset += 8

    cdaf_stats["foms"] = foms
    cdaf_stats["floating"] = cdaf_floating
    result["cdaf"] = cdaf_stats

    return result

def save_stats_dng(request, filename):
    stats = decode_stats(request.get_metadata()["PispStatsOutput"])
    zones = stats["awb"]["zones"]
    zones = [(zone["R_sum"] / zone["counted"], zone["G_sum"] / zone["counted"], zone["B_sum"] / zone["counted"]) for zone in zones]
    zones = np.array(zones)
    zones = zones.reshape(32, 32, 3)
    raw = np.zeros((64, 64), dtype=np.uint16)
    raw[0::2, 0::2] = zones[:, :, 0]
    raw[0::2, 1::2] = zones[:, :, 1]
    raw[1::2, 0::2] = zones[:, :, 1]
    raw[1::2, 1::2] = zones[:, :, 2]
    raw += request.get_metadata()["SensorBlackLevels"][0]
    config = {"format": "SRGGB16", "size": (64, 64), "stride": 64 * 2, "framesize": 64 * 64 * 2}
    model = request.picam2.camera_properties.get("Model") or "PiCamera2"
    camera = Picamera2Camera(config, request.get_metadata(), model)
    r = PICAM2DNG(camera)
    dng_compress_level = request.picam2.options.get("compress_level", 0)
    r.options(compress=dng_compress_level)
    r.convert(raw, filename)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AWB-O-Matic Tool')
    parser.add_argument('-u', '--user', help='Set the user name for saved images')
    parser.add_argument('-o', '--output', help='Override the output directory')
    parser.add_argument('--small-output', type=Path, help='Save the AWB statistics as a small DNG file in this directory (Only works on Pi 5)')
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
    if args.small_output:
        os.makedirs(args.small_output, exist_ok=True)

    print(f"User: {USER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"SSH mode: {ssh_mode}")
    print(f"Stats output directory: {args.small_output}")

    app = QApplication(sys.argv)
    window = Snapper(user=USER, output_dir=OUTPUT_DIR, ssh_mode=ssh_mode, initial_scene_id=args.initial_scene_id, small_output=args.small_output)
    window.show()
    sys.exit(app.exec_())
