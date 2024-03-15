
from max30102 import MAX30102
import hrcalc
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

class HeartRateMonitor(object):
    """
    A class that encapsulates the max30102 device into a thread
    """

    LOOP_TIME = 0.01
    
    
    def __init__(self, print_raw=False, print_result=False, filename=None):
        self.bpm = 0
        if print_raw is True:
            print('IR, Red')
        self.print_raw = print_raw
        self.print_result = print_result
        self.filename = filename

    def run_sensor(self):
        sensor = MAX30102()
        ir_data = []
        red_data = []
        bpms = []
        bpms_vec=[]
        ir_all = []
        red_all = []

        # run until told to stop
        while not self._thread.stopped:
            # check if any data is available
            num_bytes = sensor.get_data_present()
            if num_bytes > 0:
                # grab all the data and stash it into arrays
                while num_bytes > 0:
                    red, ir = sensor.read_fifo()
                    num_bytes -= 1
                    ir_data.append(ir)
                    red_data.append(red)
                    ir_all.append(ir)
                    red_all.append(red)
                    if self.print_raw:
                        print("{0}, {1}".format(ir, red))

                while len(ir_data) > 100:
                    ir_data.pop(0)
                    red_data.pop(0)

                if len(ir_data) == 100:
                    bpm, valid_bpm, spo2, valid_spo2 = hrcalc.calc_hr_and_spo2(ir_data, red_data)
                    if valid_bpm:
                        bpms.append(bpm)
                        bpms_vec.append(bpm)
                        while len(bpms) > 4:
                            bpms.pop(0)
                        self.bpm = np.mean(bpms)
                        if (np.mean(ir_data) < 50000 and np.mean(red_data) < 50000):
                            self.bpm = 0
                            if self.print_result:
                                print("Finger not detected")
                        if self.print_result:
                            print("BPM: {0}, SpO2: {1}".format(self.bpm, spo2))

            time.sleep(self.LOOP_TIME)
            
        sensor.shutdown()
        
        """fig, (ax1, ax2 )= plt.subplots(2,1)
        n = len(ir_all)
        y = [i for i in range(1, n+1)]
        ax1.scatter(y,ir_all)
        ax1.plot(y,ir_all)
        m1 = np.mean(ir_all)
        ax1.axhline(m1)
        ax1.set_ylim(m1-5000,m1+5000)
        m1med = np.median(ir_all)
        
        ax2.scatter(y,red_all)
        ax2.plot(y,red_all)
        m2 = np.mean(red_all)
        ax2.axhline(m2)
        ax2.set_ylim(m2-5000,m2+5000)
        m2med = np.median(red_all)"""
        
        combined_data = [ir_all, red_all]

        # Specify the file name
        csv_file = self.filename

        # Open the CSV file in write mode
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the data as columns
            for column_data in combined_data:
                writer.writerow(column_data)

        
        
        
        
        
        
        
        """fig, (ax1, ax2 )= plt.subplots(2,1)
        n = len(bpms_vec)
        y = [i for i in range(1, n+1)]
        ax1.scatter(y,bpms_vec)
        ax1.plot(y,bpms_vec)
        ax1.set_ylim(20,220)
        m1 = np.mean(bpms_vec)
        ax1.axhline(m1)
        m1med = np.median(bpms_vec)
        
        
        
        arr = [bpms_vec[0]]
        
        for i in bpms_vec[1:]:
            if i != arr[-1]:
                arr.append(i)
                

        n = len(arr)
        y = [i for i in range(1, n+1)]
        ax2.scatter(y,arr)
        ax2.plot(y,arr)
        ax2.set_ylim(20,220)
        m = np.mean(arr)
        ax2.axhline(m)
        plt.show()
        mmed = np.median(arr)
        
        print(m1)
        print(m1med)
        print(m)
        print(mmed)"""

    def start_sensor(self):
        self._thread = threading.Thread(target=self.run_sensor)
        self._thread.stopped = False
        self._thread.start()

    def stop_sensor(self, timeout=2.0):
        self._thread.stopped = True
        self.bpm = 0
        self._thread.join(timeout)
