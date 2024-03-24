"""Analysis_GUI  - A graphical user interface for displaying rPPG generation results using pyVHR on a Raspberry Pi or any other computer.

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



from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QGraphicsView
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
from BVP_Analysis import *
import time


        

class HeartRateAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Rate Analyzer")
        self.resize(1000,800)
        self.initUI()



    def initUI(self):
        layout = QVBoxLayout()

        self.plot_label = QLabel()
        layout.addWidget(self.plot_label)

        #self.label_sensor_status = QLabel()
        #layout.addWidget(self.label_sensor_status)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.label_distance = QLabel()
        layout.addWidget(self.label_distance)
        self.label_distance.setText("DTW Distance: No result yet")


        self.label_hr_ir = QLabel()
        layout.addWidget(self.label_hr_ir)
        self.label_hr_ir.setText("HR (ir PPG): No result yet")
        

        self.label_hr_r = QLabel()
        layout.addWidget(self.label_hr_r)
        self.label_hr_r.setText("HR (red PPG): No result yet")

        self.label_hr_rppg = QLabel()
        layout.addWidget(self.label_hr_rppg)
        self.label_hr_rppg.setText("HR (rPPG): No result yet")

        self.button_analyze = QPushButton("Analyse")
        self.button_analyze.clicked.connect(self.analyze)
        layout.addWidget(self.button_analyze)

        self.setLayout(layout)



    def analyze(self):
        
        starttime = time.time()
        
        self.figure.clear()


        new_label = "test1"
        recordingtime = 35
        method = "cpu_GREEN"
        roi_approach = 'hol'
        roi_method = "convexhull"
        pre_filt = False
        post_filt = False


        file_extension_video = ".mp4"
                
        outputlabel = new_label+ file_extension_video

        path = '/home/admin/Desktop/Main_GUI/'
        video_file = path+outputlabel



        starttime = time.time()
        #pipe = Pipeline()
        pipe = Pipeline_analysis()
        times, BPM, uncertainty, bvps = pipe.run_on_video(video_file,cuda=False, roi_approach=roi_approach, roi_method=roi_method,pre_filt=pre_filt, post_filt=post_filt, method = method)
        endtime = time.time()
        evaluation = endtime-starttime

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

        print(beat_ir)
        print(beat_r)
        print(beat_rppg)

        def HR_calc(leng, time):
            HR =(60/time) *leng
            return HR

        HR_ir = HR_calc(beat_ir, recordingtime-1.0)
        HR_r = HR_calc(beat_r, recordingtime-1.0)
        HR_rppg = HR_calc(beat_rppg, recordingtime-1.0)

        self.label_hr_ir.setText(f"HR (ir PPG): {round(HR_ir, 2)}")
        self.label_hr_r.setText(f"HR (red PPG): {round(HR_r, 2)}")
        self.label_hr_rppg.setText(f"HR (rPPG): {round(HR_rppg, 2)}")
        
        endtime = time.time()
        evalv = endtime-starttime
        
        print("Done! ")
        print(evalv)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeartRateAnalyzer()
    window.show()
    sys.exit(app.exec_())
