from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QGraphicsView, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage  
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch


#from pyVHR.analysis.pipeline import Pipeline
from BVP_Analysis_Multiprocessing import run_on_video as rov
from BVP_Analysis_Multiprocessing import *
import time
import numpy as np


        

class HeartRateAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Rate Analyzer")
        self.resize(1000,1400)
        self.initUI()



    def initUI(self):
        layout = QVBoxLayout()

        #self.button_choose_file = QPushButton("Choose File")
        #self.button_choose_file.clicked.connect(self.choose_file)
        #layout.addWidget(self.button_choose_file)

        self.textbox_file_name = QLineEdit()
        layout.addWidget(self.textbox_file_name)

        self.plot_label = QLabel()
        layout.addWidget(self.plot_label)

        #self.label_sensor_status = QLabel()
        #layout.addWidget(self.label_sensor_status)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)
        layout.addWidget(self.canvas2) 

        self.label_distance = QLabel()
        layout.addWidget(self.label_distance)
        self.label_distance.setText("DTW Distance: No result yet")

        self.label_lux = QLabel()
        layout.addWidget(self.label_lux)
        self.label_lux.setText("LUX: No result yet")

        self.label_hr_ir = QLabel()
        layout.addWidget(self.label_hr_ir)
        self.label_hr_ir.setText("ir PPG: No result yet")
        

        self.label_hr_r = QLabel()
        layout.addWidget(self.label_hr_r)
        self.label_hr_r.setText("red PPG: No result yet")

        self.label_hr_rppg = QLabel()
        layout.addWidget(self.label_hr_rppg)
        self.label_hr_rppg.setText("rPPG: No result yet")

        self.button_analyze = QPushButton("Analyse")
        self.button_analyze.clicked.connect(self.analyze)
        layout.addWidget(self.button_analyze)

        

        self.setLayout(layout)



    def choose_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Video File", "", "Video Files (*.mp4);;All Files (*)", options=options)
        if file_name:
            self.file_path = file_name

    def analyze(self):

        self.figure.clear()

        file_name = self.textbox_file_name.text()
        if not file_name:
            print("Please enter the name of the file.")
            return

        #if not self.file_path:
        #    print("Please choose a file first.")
        #    return
        

        new_label = file_name
        recordingtime = 35
        method ="cpu_LGI" #"cpu_LGI_TSVD"
        roi_approach = 'hol'
        roi_method = "convexhull"
        pre_filt = False
        post_filt = False


        file_extension_video = ".mp4"
                
        outputlabel = new_label + file_extension_video

        

        path = 'path'
        video_file = path+outputlabel

        lux_val = main_brightness(video_file)

        starttime = time.time()
        times, BPM, uncertainty, bvps = rov(video_file,cuda=False, roi_approach=roi_approach, roi_method=roi_method,pre_filt=pre_filt, post_filt=post_filt, method = method)
        endtime = time.time()
        evaluation = endtime-starttime
        print("Zeit für berechnung", evaluation)
        HRmean= np.mean(BPM)
        print("HR mean rPPG and PPG")
        print(HRmean)

        output2=".csv"
        #path = "/home/admin/Desktop/pyVHR-pyVHR_CPU_gui/"
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



        def HR_calc(leng, time):
            HR =(60/time) *leng
            return HR

        HR_ir = HR_calc(beat_ir, recordingtime-1.0)
        HR_r = HR_calc(beat_r, recordingtime-1.0)
        HR_rppg = HR_calc(beat_rppg, recordingtime-1.0)

        

        

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
        plt.axvline(x=max_freq_y1, color='red', linestyle='--', label=f'Max amplitude for PPG: {max_freq_y1:.2f}')
        plt.axvline(x=max_freq_y2, color='blue', linestyle='--', label=f'Max amplitude for rPPG: {max_freq_y2:.2f}')
        distance = np.abs(max_freq_y1 - max_freq_y2)
        #plt.text(0.5, 0.9, f'Distance between max frequencies: {distance:.2f}', transform=plt.gca().transAxes)

        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        plt.title('FFT ', fontsize=16)
        plt.xlim(0, 4) 
        #plt.ylim(0, 100)  # Set limit for y-axis
        plt.legend()
        plt.savefig('fft_plot.png')
        self.canvas2.draw()

        fre_HR_red = max_freq_y1*60
        fre_HR_rppg = max_freq_y2*60

        fdiff = np.abs(max_freq_y2-max_freq_y1)
        print(fdiff)

        self.label_lux.setText(f" Lux: {lux_val} ")
        self.label_hr_ir.setText(f" ir PPG    - HR : {round(HR_ir, 2)}   - Peaks found: {beat_ir} ")
        self.label_hr_r.setText(f" red PPG - HR : {round(HR_r, 2)}   - Peaks found: {beat_r} - Freq HR: {fre_HR_red:.2f}")
        self.label_hr_rppg.setText(f" rPPG      - HR : {round(HR_rppg, 2)}   - Peaks found: {beat_rppg} - Freq HR: {fre_HR_rppg:.2f}")

        print("Done! ")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeartRateAnalyzer()
    window.show()
    sys.exit(app.exec_())
