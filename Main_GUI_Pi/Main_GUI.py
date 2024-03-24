
"""Main_GUI_Pi  - A graphical user interface for displaying rPPG generation results using pyVHR on a Raspberry Pi.

Copyright (C) 2024 morijx

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>."""


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit, QComboBox, QCheckBox, QGraphicsView, QMessageBox
import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from heartrate_monitor import HeartRateMonitor
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from pyVHR.analysis.pipeline import Pipeline
import plotly.express as px
import numpy as np
import argparse
import time
import csv

import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, cheby2
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import ConnectionPatch

from BVP_Analysis import *

import cv2

class MyGUI(QWidget):
    
    update_sensor_status_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(800,1000)
        self.show_error_message()
        self.setWindowTitle("HR")
        # Initialize the GUI elements
        self.init_ui()
        
        self.PPG = 0
        self.rPPG =0
        self.camera = None

    def init_ui(self):
        # Create labels
        self.label_label = QLabel("Label")

        # Create line edits for user input
        self.edit_label = QLineEdit(self)
        self.edit_label.setPlaceholderText("Enter label name")
        self.edit_label.setAlignment(Qt.AlignCenter)
        self.edit_label.setText("001")
        
        self.label_rectime = QLabel("Recording time (seconds)")
        self.edit_recordingtime = QLineEdit(self)
        self.edit_recordingtime.setPlaceholderText("Enter recording time (seconds)")
        self.edit_recordingtime.setAlignment(Qt.AlignCenter)
        self.edit_recordingtime.setText("35")

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)
        


        # Create dynamic labels for heart rates
        self.label_distance = QLabel()
        self.label_distance.setText("DTW Distance: No result yet")


        self.label_hr_ir = QLabel()
        self.label_hr_ir.setText("HR (ir PPG): No result yet")
        

        self.label_hr_r = QLabel()
        self.label_hr_r.setText("HR (red PPG): No result yet")

        self.label_hr_rppg = QLabel()
        self.label_hr_rppg.setText("HR (rPPG): No result yet")

        self.label_sensor_status = QLabel("Status: Not Started")

        # Create a button
        self.button_start = QPushButton("Start Program")
        self.button_start.clicked.connect(self.start_program)

        # Create dynamic labels

        self.plot_view = QGraphicsView()

        # Arrange the elements in a layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_label)
        layout.addWidget(self.edit_label)
        layout.addWidget(self.label_rectime)
        layout.addWidget(self.edit_recordingtime)
        #layout.addWidget(self.combo_method)

        layout.addWidget(self.button_start)
        layout.addWidget(self.canvas)
        layout.addWidget(self.canvas2)
        #layout.addWidget(self.label_sensor_status)
        
        #layout.addWidget(self.plot_view)
        layout.addWidget(self.label_distance)
        layout.addWidget(self.label_hr_ir)
        layout.addWidget(self.label_hr_r)
        layout.addWidget(self.label_hr_rppg)
        
        

        # Set the layout for the main window
        self.setLayout(layout)

    def show_error_message(self):
        QMessageBox.critical(self, "Error", "This program comes with ABSOLUTELY NO WARRANT")



    def start_program(self):
        # Get the text from the QLineEdit

        new_label = self.edit_label.text()
        recordingtime = int(self.edit_recordingtime.text())
        method = "cpu_GREEN"
        roi_approach = 'hol'
        roi_method = "convexhull"
        pre_filt = False
        post_filt = False
        
        self.figure.clear()


        if self.camera is None:
            self.camera = Picamera2()
            modes = self.camera.sensor_modes
            mode = modes[1]
            config = self.camera.create_video_configuration(sensor={'output_size': mode['size']})
            self.camera.video_configuration.size = (640,480)
            self.camera.configure(config)
            


        time.sleep(3)

                    
        parser = argparse.ArgumentParser(description="Read and print data from MAX30102")
        parser.add_argument("-r", "--raw", action="store_true",
                            help="print raw data instead of calculation result")
        parser.add_argument("-t", "--time", type=int, default=recordingtime*2,
                            help="duration in seconds to read from sensor, default 30")
        args = parser.parse_args()

        print('sensor starting...')
        self.label_sensor_status.setText("Process starting...")
        #self.update_signal.emit("Sensor starting... \n")
        file_extension_csv = ".csv"
        filename = new_label+ file_extension_csv
        hrm = HeartRateMonitor(print_raw=args.raw, print_result=(not args.raw), filename=filename)
        hrm.start_sensor()
        
        file_extension_video = ".mp4"
        
        outputlabel = new_label+ file_extension_video
        
        
        print(self.camera.sensor_modes)
        encoder = H264Encoder(bitrate=1700000)
        output = FfmpegOutput(outputlabel, audio=False)

        self.camera.start_recording(encoder, output)
        time.sleep(recordingtime)
        
        self.camera.stop_recording()
        
        hrm.stop_sensor()
        print('sensor stoped!')
        self.label_sensor_status.setText("Sensor stoped")
        
        time.sleep(0.5)
        
        #self.update_signal.emit("Sensor Stopped!\n")
        path = "/home/admin/Desktop/Main_GUI/"
        video_file = path+outputlabel

        starttime = time.time()
        #pipe = Pipeline()
        pipe = Pipeline_analysis()
        times, BPM, uncertainty, bvps = pipe.run_on_video(video_file,cuda=False, roi_approach=roi_approach, roi_method=roi_method,pre_filt=pre_filt, post_filt=post_filt, method = method)
        endtime = time.time()
        evaluation = endtime-starttime
        self.label_sensor_status.setText(f"Time to evaluate the HR: {evaluation}")


        HRmean= np.mean(BPM)
        print("HR mean rPPG and PPG")
        print(HRmean)

        output2=".csv"
        csv_file = path+new_label+output2

        peaks1, peaks2, peaks_rppg, y1, y2, x1, x2, path_norm, dist = analysis(video_file, csv_file, bvps)

        self.label_distance.setText(f"Distance: {dist}")

        series_line_options = {}
        warping_line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}

        self.figure.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
        axs = self.figure.axes

        axs[0].plot(x1, y1, **series_line_options)
        axs[1].plot(x2, y2, **series_line_options)

        axs[0].set_xlabel('')
        axs[0].set_ylabel('Intensity red PPG')

        axs[1].set_ylabel('Intensity rPPG')
        axs[1].set_xlabel('Time [s]')

        self.figure.suptitle('DTW red PPG and rPPG')

        for r_c, c_c in path_norm:
            if r_c >= 0 and c_c >= 0:
                con = ConnectionPatch(xyA=[x1[r_c], y1[r_c]], coordsA=axs[0].transData,
                                    xyB=[x2[c_c], y2[c_c]], coordsB=axs[1].transData, **warping_line_options)
                self.figure.add_artist(con)

        plt.tight_layout()

        self.canvas.draw()

        beat_ir = len(peaks1)
        beat_r = len(peaks2)
        beat_rppg = len(peaks_rppg)

        print(beat_ir)
        print(beat_r)
        print(beat_rppg)

        def HR_calc(leng, time):
            HR =(60/time) *leng
            return HR

        HR_ir = HR_calc(beat_ir, recordingtime-0.8)
        HR_r = HR_calc(beat_r, recordingtime-0.8)
        HR_rppg = HR_calc(beat_rppg, recordingtime-0.8)

        y1 = y1 - np.mean(y1)
        fft_y1 = np.fft.fft(y1)
        frequencies_y1 = np.fft.fftfreq(len(fft_y1), d=(1/25 ))  # Frequency values


        positive_freq_mask = frequencies_y1 >= 0
        frequencies_y1 = frequencies_y1[positive_freq_mask]
        fft_y1 = fft_y1[positive_freq_mask]

        y2 = y2 - np.mean(y2)
        fft_y2 = np.fft.fft(y2)
        frequencies_y2 = np.fft.fftfreq(len(fft_y2), d=(1/30 ))  # Frequency values


        positive_freq_mask = frequencies_y2 >= 0
        frequencies_y2 = frequencies_y2[positive_freq_mask]
        fft_y2 = fft_y2[positive_freq_mask]

        max_index_y1 = np.argmax(np.abs(fft_y1))
        max_index_y2 = np.argmax(np.abs(fft_y2))
        max_freq_y1 = frequencies_y1[max_index_y1]
        max_freq_y2 = frequencies_y2[max_index_y2]

        # Plot FFT
        plt.figure(self.figure2.number)
        plt.clf()
        plt.plot(frequencies_y1, np.abs(fft_y1), label='PPG (red)')
        plt.plot(frequencies_y2, np.abs(fft_y2), label='rPPG')
        plt.axvline(x=max_freq_y1, color='red', linestyle='--', label=f'Max amplitude for y1: {max_freq_y1:.2f}')
        plt.axvline(x=max_freq_y2, color='blue', linestyle='--', label=f'Max amplitude for y2: {max_freq_y2:.2f}')
        distance = np.abs(max_freq_y1 - max_freq_y2)
        #plt.text(0.5, 0.9, f'Distance between max frequencies: {distance:.2f}', transform=plt.gca().transAxes)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT ')
        plt.xlim(0, 4) 
        #plt.ylim(0, 100)  # Set limit for y-axis
        plt.legend()
        self.canvas2.draw()

        fre_HR_red = max_freq_y1*60
        fre_HR_rppg = max_freq_y2*60

        self.label_hr_ir.setText(f" ir PPG    - HR : {round(HR_ir, 2)}   - Peaks found: {beat_ir} ")
        self.label_hr_r.setText(f" red PPG - HR : {round(HR_r, 2)}   - Peaks found: {beat_r} - Freq HR: {fre_HR_red:.2f}")
        self.label_hr_rppg.setText(f" rPPG      - HR : {round(HR_rppg, 2)}   - Peaks found: {beat_rppg} - Freq HR: {fre_HR_rppg:.2f}")


if __name__ == '__main__':
    app = QApplication([])
    window = MyGUI()
    window.show()
    app.exec_()



