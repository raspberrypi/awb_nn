#! /usr/bin/env python3

import numpy as np
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QScrollArea, QHBoxLayout,
                            QVBoxLayout, QLineEdit, QLabel, QSlider)
from PyQt5.QtGui import QPixmap, QWheelEvent, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QRect

# Constants
MIN_COLOUR_TEMP = 2800
MAX_COLOUR_TEMP = 7000

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_start = None
        self.selection_end = None
        self.setAlignment(Qt.AlignCenter)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_start and self.selection_end:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            rect = QRect(self.selection_start, self.selection_end).normalized()
            painter.drawRect(rect)

class ImageDialog(QDialog):
    def __init__(self, parent=None, dng=None, colour_gains=None, box=None):
        super().__init__(parent)
        self.setWindowTitle("Annotate Image")
        self.setModal(True)
        self.setGeometry(50, 50, 1200, 900)  # Increased dialog size

        # Add property to store the selected rectangle
        self.selected_rect = None
        self.MIN_SIZE = 10  # Minimum size for selection in pixels

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.setLayout(layout)

        # Create scroll area for panning
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)  # Remove frame
        layout.addWidget(self.scroll_area)

        # Create label for displaying image
        self.image_label = ImageLabel()
        self.scroll_area.setWidget(self.image_label)

        # Add instructions
        instructions = QLabel("Click and drag to pan. Mouse wheel to zoom. Ctrl+Click and drag to set grey rectangle\n"
                              "Alternatively set the colour gains or drag the colour temperature slider directly\n"
                              "Click the Accept button to finish")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        # Add colour gains display
        gains_layout = QHBoxLayout()
        gains_layout.setContentsMargins(20, 0, 0, 0)  # Add left margin
        gains_layout.addWidget(QLabel("Colour Gains:"))

        self.r_gain_edit = QLineEdit()
        self.r_gain_edit.setPlaceholderText("Red")
        self.r_gain_edit.setFixedWidth(80)
        self.r_gain_edit.setStyleSheet("background-color: white; padding: 5px; border: 1px solid #ccc;")
        self.r_gain_edit.returnPressed.connect(self.on_gain_changed)

        self.b_gain_edit = QLineEdit()
        self.b_gain_edit.setPlaceholderText("Blue")
        self.b_gain_edit.setFixedWidth(80)
        self.b_gain_edit.setStyleSheet("background-color: white; padding: 5px; border: 1px solid #ccc;")
        self.b_gain_edit.returnPressed.connect(self.on_gain_changed)

        gains_layout.addWidget(QLabel("Red ="))
        gains_layout.addWidget(self.r_gain_edit)
        gains_layout.addWidget(QLabel("Blue ="))
        gains_layout.addWidget(self.b_gain_edit)
        gains_layout.addStretch()

        layout.addLayout(gains_layout)

        # Add colour temperature slider
        temp_layout = QHBoxLayout()
        temp_layout.setContentsMargins(20, 0, 0, 0)  # Add left margin
        temp_layout.addWidget(QLabel("Colour Temp:"))

        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(MIN_COLOUR_TEMP)
        self.temp_slider.setMaximum(MAX_COLOUR_TEMP)
        self.temp_slider.setValue(int(MIN_COLOUR_TEMP))  # Will be updated when colour_temp is set
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        self.temp_slider.setTickInterval(500)
        self.temp_slider.valueChanged.connect(self.on_temp_slider_moved)
        self.temp_slider.sliderReleased.connect(self.on_temp_slider_changed)

        self.temp_label = QLabel(f"{MIN_COLOUR_TEMP}K")
        self.temp_label.setFixedWidth(80)
        self.temp_label.setStyleSheet("background-color: white; padding: 5px; border: 1px solid #ccc;")

        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_label)
        temp_layout.addStretch()

        layout.addLayout(temp_layout)

        # Add transverse slider (below temperature slider)
        transverse_layout = QHBoxLayout()
        transverse_layout.setContentsMargins(20, 0, 0, 0)
        transverse_layout.addWidget(QLabel("Purple/Green:"))

        self.transverse_slider = QSlider(Qt.Horizontal)
        self.transverse_slider.setMinimum(-25)
        self.transverse_slider.setMaximum(25)
        self.transverse_slider.setValue(0)
        self.transverse_slider.setTickPosition(QSlider.TicksBelow)
        self.transverse_slider.setTickInterval(5)
        self.transverse_slider.valueChanged.connect(self.on_transverse_slider_moved)
        self.transverse_slider.sliderReleased.connect(self.on_transverse_slider_changed)

        self.transverse_value_label = QLabel("0.00")
        self.transverse_value_label.setFixedWidth(80)
        self.transverse_value_label.setStyleSheet("background-color: white; padding: 5px; border: 1px solid #ccc;")

        transverse_layout.addWidget(self.transverse_slider)
        transverse_layout.addWidget(self.transverse_value_label)
        transverse_layout.addStretch()

        layout.addLayout(transverse_layout)

        # Add Done button in a centered layout
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)  # Add some margin around the button
        button_layout.addStretch(1)
        button_box = QDialogButtonBox()
        self.accept_button = button_box.addButton("Accept", QDialogButtonBox.AcceptRole)
        self.accept_button.clicked.connect(self.accept)
        cancel_button = button_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(button_box)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        # Initialize zoom and pan variables
        self.zoom_factor = 1.0
        self.min_zoom_factor = 1.0
        self.pan_start = QPoint()
        self.panning = False
        self.original_pixmap = None
        self.backup_pixmap = None

        # Initialize selection rectangle variables
        self.is_selecting = False
        self.ctrl_pressed = False

        # Enable mouse tracking for panning
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent
        self.image_label.wheelEvent = self.wheelEvent
        self.image_label.leaveEvent = self.leaveEvent

        # Create RGB info label
        self.rgb_info_label = QLabel()
        self.rgb_info_label.setStyleSheet("background-color: rgba(0, 0, 0, 0.8); color: white; padding: 5px; border: 1px solid white; border-radius: 3px;")
        self.rgb_info_label.setMinimumWidth(60)
        self.rgb_info_label.setAlignment(Qt.AlignCenter)
        self.rgb_info_label.hide()
        self.rgb_info_label.setWindowFlags(Qt.ToolTip)
        self.rgb_info_label.raise_()

        self.dng = dng
        self.colour_gains = colour_gains
        self.box = box  # Store the box for later use
        if box is not None:
            self.box_to_colour_gains(box)
        else:
            self.update_colour_temp()
        self.develop_image()
        self.update_image()
        self.update_colour_gains_display()

    def box_to_colour_gains(self, box):
        for i in range(3):
            self.dng.restore()
            self.dng.do_digital_gain(0.25)
            if i:
                self.dng.do_lsc(self.colour_temp)
            R_ave, G_ave, B_ave = self.dng.rgb_averages(*box)
            print("Averages: R", R_ave, "G", G_ave, "B", B_ave)
            colour_gains = (G_ave / R_ave, G_ave / B_ave)
            self.colour_gains = (round(colour_gains[0], 3), round(colour_gains[1], 3))
            self.update_colour_temp()
            print("Calculated colour gains:", self.colour_gains, "colour temp:", self.colour_temp)

    def update_colour_temp(self):
        # Set the colour temp from the gains, but we clamp it to our allowable range.
        red_blue = (1.0 / self.colour_gains[0], 1.0 / self.colour_gains[1])
        self.colour_temp, self.transverse_value = self.dng.tuning.colour_temp_curve.invert_with_transverse_multiple(red_blue)
        self.transverse_slider.setValue(int(self.transverse_value * 100))
        self.transverse_value_label.setText(f"{self.transverse_value:.3f}")
        if self.colour_temp < MIN_COLOUR_TEMP:
            self.colour_temp = MIN_COLOUR_TEMP
            colour_gains = 1.0 / self.dng.tuning.get_colour_values(MIN_COLOUR_TEMP)
            self.colour_gains = (round(colour_gains[0], 3), round(colour_gains[1], 3))
        elif self.colour_temp > MAX_COLOUR_TEMP:
            self.colour_temp = MAX_COLOUR_TEMP
            colour_gains = 1.0 / self.dng.tuning.get_colour_values(MAX_COLOUR_TEMP)
            self.colour_gains = (round(colour_gains[0], 3), round(colour_gains[1], 3))
        print("Updated colour_temp to", self.colour_temp, " and gains to", self.colour_gains)
        self.update_colour_gains_display()

    def update_colour_gains_display(self):
        """Update the colour gains display fields and temperature slider"""
        if hasattr(self, 'colour_gains') and self.colour_gains is not None:
            r_gain, b_gain = self.colour_gains
            self.r_gain_edit.setText(f"{r_gain:.3f}")
            self.b_gain_edit.setText(f"{b_gain:.3f}")

            # Update temperature slider and label if colour_temp is available
            if hasattr(self, 'colour_temp') and self.colour_temp is not None:
                self.temp_slider.setValue(int(self.colour_temp))
                self.temp_label.setText(f"{self.colour_temp}K")

            if hasattr(self, 'transverse_value') and self.transverse_value is not None:
                self.transverse_slider.setValue(int(self.transverse_value * 100))
                self.transverse_value_label.setText(f"{self.transverse_value:.3f}")
        else:
            self.r_gain_edit.setText("")
            self.b_gain_edit.setText("")

    def on_gain_changed(self):
        """Handle changes to the colour gain input fields"""
        try:
            r_gain_text = self.r_gain_edit.text().strip()
            b_gain_text = self.b_gain_edit.text().strip()

            if r_gain_text and b_gain_text:
                r_gain = float(r_gain_text)
                b_gain = float(b_gain_text)

                # Update the colour gains
                self.colour_gains = (r_gain, b_gain)

                # Clear rectangle selection
                self.image_label.selection_start = None
                self.image_label.selection_end = None
                self.image_label.update()

                # Disable Accept button while updating
                self.accept_button.setEnabled(False)
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents() # when the button is re-enabled, that means it's finished

                # Update colour temperature and redevelop image
                self.update_colour_temp()
                self.develop_image()
                self.update_image()

                # Enable Accept button since we have valid changes
                self.accept_button.setEnabled(True)

        except ValueError:
            # Invalid input, ignore
            pass

    def on_temp_slider_changed(self):
        """Handle changes to the temperature slider"""
        # Get the current slider value
        value = self.temp_slider.value()

        # Update the temperature label
        self.temp_label.setText(f"{value}K")

        # Update colour temperature
        self.colour_temp = value

        # Clear rectangle selection
        self.image_label.selection_start = None
        self.image_label.selection_end = None
        self.image_label.update()

        # Disable Accept button while updating
        self.accept_button.setEnabled(False)
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents() # when the button is re-enabled, that means it's finished

        # Get new colour gains from the tuning
        red_blue = self.dng.tuning.get_colour_values(value)
        red_blue = red_blue + self.transverse_value * self.dng.tuning.colour_temp_curve.transverse(value)
        self.colour_gains = (round(1.0 / red_blue[0], 3), round(1.0 / red_blue[1], 3))

        # Update the gain input fields
        self.update_colour_gains_display()

        # Redevelop the image with new values
        self.develop_image()
        self.update_image()

        # Enable Accept button since we have valid changes
        self.accept_button.setEnabled(True)

    def on_temp_slider_moved(self, value):
        """Handle slider movement - just update the label, don't process the image"""
        # Update the temperature label in real-time
        self.temp_label.setText(f"{value}K")

    def on_transverse_slider_changed(self):
        """Update the transverse value label when the slider changes."""
        value = self.transverse_slider.value()
        self.transverse_value_label.setText(f"{value / 100:.2f}")
        self.transverse_value = value / 100

        self.image_label.selection_start = None
        self.image_label.selection_end = None
        self.image_label.update()

        self.accept_button.setEnabled(False)
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        # Get new colour gains from the tuning
        red_blue = self.dng.tuning.get_colour_values(self.colour_temp)
        red_blue = red_blue + self.transverse_value * self.dng.tuning.colour_temp_curve.transverse(self.colour_temp)
        self.colour_gains = (round(1.0 / red_blue[0], 3), round(1.0 / red_blue[1], 3))

        # Update the gain input fields
        self.update_colour_gains_display()

        # Redevelop the image with new values
        self.develop_image()
        self.update_image()

        # Enable Accept button since we have valid changes
        self.accept_button.setEnabled(True)


    def on_transverse_slider_moved(self, value):
        """Handle slider movement - just update the label, don't process the image"""
        self.transverse_value_label.setText(f"{value / 100:.2f}")

    def show_rgb_info(self, pos):
        """Show RGB values at the given position"""
        if not hasattr(self, 'original_pixmap') or self.original_pixmap is None:
            return

        # Convert viewport position to image position
        image_x = int(pos.x() / self.zoom_factor)
        image_y = int(pos.y() / self.zoom_factor)

        # Check if position is within image bounds
        if (0 <= image_x < self.original_pixmap.width() and
            0 <= image_y < self.original_pixmap.height()):

            # Get the image data
            image = self.original_pixmap.toImage()

            # Calculate 16x16 area bounds
            half_size = 8
            start_x = max(0, image_x - half_size)
            end_x = min(self.original_pixmap.width(), image_x + half_size)
            start_y = max(0, image_y - half_size)
            end_y = min(self.original_pixmap.height(), image_y + half_size)

            # Calculate average RGB values for the 16x16 area
            total_r = total_g = total_b = 0
            pixel_count = 0

            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    pixel_color = image.pixelColor(x, y)
                    total_r += pixel_color.red()
                    total_g += pixel_color.green()
                    total_b += pixel_color.blue()
                    pixel_count += 1

            # Calculate averages
            avg_r = int(total_r / pixel_count) if pixel_count > 0 else 0
            avg_g = int(total_g / pixel_count) if pixel_count > 0 else 0
            avg_b = int(total_b / pixel_count) if pixel_count > 0 else 0

            # Update the label text
            self.rgb_info_label.setText(f"R: {avg_r}\nG: {avg_g}\nB: {avg_b}")

            # Position the label near the mouse cursor
            global_pos = self.image_label.mapToGlobal(pos)
            label_x = global_pos.x() + 15
            label_y = global_pos.y() - 50

            # Show the label
            self.rgb_info_label.move(label_x, label_y)
            self.rgb_info_label.show()
        else:
            # Hide the label if outside image bounds
            self.rgb_info_label.hide()

    def leaveEvent(self, event):
        """Hide RGB info when mouse leaves the image area"""
        self.rgb_info_label.hide()

    def develop_image(self):
        print("Develop image with colour gains", self.colour_gains, "and colour temp", self.colour_temp)
        self.dng.restore()
        self.dng.do_lsc(self.colour_temp)
        rgb_arr = self.dng.convert(colour_gains=self.colour_gains, median_filter_passes=0)

        # Convert back to QPixmap
        height, width = rgb_arr.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(rgb_arr.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        self.set_image(QPixmap.fromImage(q_img))

    def draw_box_rectangle(self):
        """Draw the rectangle from the provided box coordinates"""
        if self.box is None:
            return

        # Convert image coordinates to viewport coordinates
        x0, y0, x1, y1 = self.box

        # Convert to viewport coordinates using current zoom factor
        start_x = int(x0 * self.zoom_factor)
        start_y = int(y0 * self.zoom_factor)
        end_x = int(x1 * self.zoom_factor)
        end_y = int(y1 * self.zoom_factor)

        # Set the selection points in the image label
        self.image_label.selection_start = QPoint(start_x, start_y)
        self.image_label.selection_end = QPoint(end_x, end_y)

        # Update the display
        self.image_label.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctrl_pressed = True
            self.image_label.setCursor(Qt.CrossCursor)
            # Clear previous selection when Ctrl is pressed
            self.image_label.selection_start = None
            self.image_label.selection_end = None
            self.image_label.update()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Prevent Enter key from accepting the dialog
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctrl_pressed = False
            self.image_label.setCursor(Qt.ArrowCursor)
            # Don't clear the selection when Ctrl is released
        super().keyReleaseEvent(event)

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        # Create a backup copy of the original pixmap
        self.backup_pixmap = QPixmap(pixmap)
        # We'll calculate the zoom factor in showEvent

    def showEvent(self, event):
        super().showEvent(event)
        if self.original_pixmap is not None:
            self.update_min_zoom_factor()
            self.zoom_factor = self.min_zoom_factor
            self.update_image()

            # Draw the rectangle if a box was provided
            if hasattr(self, 'box') and self.box is not None:
                self.draw_box_rectangle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap is not None:
            self.update_min_zoom_factor()
            # If current zoom is less than min, update to min
            if self.zoom_factor < self.min_zoom_factor:
                self.zoom_factor = self.min_zoom_factor
                self.update_image()

    def update_min_zoom_factor(self):
        # Calculate initial zoom factor to fill the window
        viewport_size = self.scroll_area.viewport().size()

        # Calculate zoom factors for width and height
        width_ratio = viewport_size.width() / self.original_pixmap.width()
        height_ratio = viewport_size.height() / self.original_pixmap.height()

        # Use the larger ratio to fill the window
        self.min_zoom_factor = max(width_ratio, height_ratio)

    def update_image(self):
        # Scale the image based on current zoom factor
        scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()

    def wheelEvent(self, event: QWheelEvent):
        # Clear rectangle when zooming
        self.image_label.selection_start = None
        self.image_label.selection_end = None
        self.image_label.update()

        # Get current scroll positions
        old_h_scroll = self.scroll_area.horizontalScrollBar().value()
        old_v_scroll = self.scroll_area.verticalScrollBar().value()

        # Get mouse position relative to the viewport
        mouse_pos = event.pos()

        # Calculate position relative to the image
        image_pos = QPoint(
            mouse_pos.x() + old_h_scroll,
            mouse_pos.y() + old_v_scroll
        )

        # Calculate the position as a ratio of the image size
        image_size = self.image_label.size()
        pos_ratio_x = image_pos.x() / image_size.width()
        pos_ratio_y = image_pos.y() / image_size.height()

        # Zoom in/out with mouse wheel
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        else:
            self.zoom_factor *= 0.9  # Zoom out

        # Limit zoom range
        self.zoom_factor = max(self.min_zoom_factor, min(5.0, self.zoom_factor))
        self.update_image()

        # Calculate new image size
        new_image_size = self.image_label.size()

        # Calculate the new scroll positions to keep the same pixel under the mouse
        new_h_scroll = int(pos_ratio_x * new_image_size.width() - mouse_pos.x())
        new_v_scroll = int(pos_ratio_y * new_image_size.height() - mouse_pos.y())

        # Ensure scroll positions are within valid range
        h_scroll_bar = self.scroll_area.horizontalScrollBar()
        v_scroll_bar = self.scroll_area.verticalScrollBar()

        new_h_scroll = max(0, min(new_h_scroll, h_scroll_bar.maximum()))
        new_v_scroll = max(0, min(new_v_scroll, v_scroll_bar.maximum()))

        # Set new scroll positions
        h_scroll_bar.setValue(new_h_scroll)
        v_scroll_bar.setValue(new_v_scroll)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.ctrl_pressed:
                self.is_selecting = True
                self.image_label.selection_start = event.pos()
                self.image_label.selection_end = event.pos()
            else:
                # Clear rectangle when starting to pan
                self.image_label.selection_start = None
                self.image_label.selection_end = None
                self.image_label.update()
                self.pan_start = event.pos()
                self.panning = True
                self.image_label.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.ctrl_pressed and self.is_selecting:
            self.image_label.selection_end = event.pos()
            self.image_label.update()
        elif self.panning:
            delta = event.pos() - self.pan_start
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            self.pan_start = event.pos()

        # Show RGB values under mouse cursor
        self.show_rgb_info(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.ctrl_pressed:
                self.is_selecting = False
                if self.image_label.selection_start and self.image_label.selection_end:
                    # Convert selection coordinates to image coordinates
                    # The selection coordinates are already in the viewport's coordinate space,
                    # so we just need to divide by zoom_factor to get back to original image coordinates
                    start_x = int(self.image_label.selection_start.x() / self.zoom_factor)
                    start_y = int(self.image_label.selection_start.y() / self.zoom_factor)
                    end_x = int(self.image_label.selection_end.x() / self.zoom_factor)
                    end_y = int(self.image_label.selection_end.y() / self.zoom_factor)

                    # Create rectangle in original image coordinates
                    rect = QRect(
                        min(start_x, end_x),
                        min(start_y, end_y),
                        abs(end_x - start_x),
                        abs(end_y - start_y)
                    )

                    # Only accept selection if it's large enough
                    if rect.width() >= self.MIN_SIZE and rect.height() >= self.MIN_SIZE:
                        # Store the rectangle in original image coordinates
                        self.selected_rect = {
                            'x': rect.x(),
                            'y': rect.y(),
                            'width': rect.width(),
                            'height': rect.height()
                        }
                        print("-" * 40)
                        print(f"Selected rectangle:", self.selected_rect)
                        self.accept_button.setEnabled(False)
                        from PyQt5.QtWidgets import QApplication
                        QApplication.processEvents() # when the button is re-enabled, that means it's finished

                        self.box_to_colour_gains((rect.x(), rect.y(),
                                                rect.x() + rect.width(), rect.y() + rect.height()))
                        self.develop_image()
                        self.update_image()

                        self.accept_button.setEnabled(True)  # Enable Accept button when valid selection is made
                    else:
                        # Clear the selection if it's too small
                        self.image_label.selection_start = None
                        self.image_label.selection_end = None
                        self.image_label.update()
                        self.accept_button.setEnabled(False)  # Disable Accept button when selection is cleared
            else:
                self.panning = False
                self.image_label.setCursor(Qt.ArrowCursor)

    def on_cancel(self):
        """Handle cancel button click by clearing selection and closing dialog"""
        self.image_label.selection_start = None
        self.image_label.selection_end = None
        self.image_label.update()
        self.selected_rect = None
        self.accept_button.setEnabled(False)  # Disable Accept button when canceling
        self.reject()