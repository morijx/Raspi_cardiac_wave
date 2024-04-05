# MAIN_GUI_PI

The files heartrate_monitor, hrcalc and max30102 were not changed and just taken as is form the Kiosk at BMHT ETH Zürich.
The BVP_Analysis file contains the Pipeline form pyVHR with slight changes to return the BVP for further analysis.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pyVHR 2.0](https://img.shields.io/badge/pyVHR-2.0-blue.svg)](https://pypi.org/project/pyVHR/)
![GitHub last commit](https://img.shields.io/github/last-commit/morijx/Raspi_cardiac_wave)
![GitHub license](https://img.shields.io/github/license/morijx/Raspi_cardiac_wave)
![Multiproccessing NO](https://img.shields.io/badge/Multiprocessing-NO-red.svg)


## Use
To utilize the graphical user interface (GUI), execute Main_GUI.py on the Raspberry Pi. Ensure that all the required code files are located in the same directory as Main_GUI.py. If encountering any issues, double-check the GPIO pin configurations and verify that the selected camera mode is supported.

For data analysis on a computer, launch Analysis_GUI.py along with the accompanying files stored in the same directory.

## Results
In the code the DTW is analysed and the distance calculated.However, a comparison of the distances with other values from the literature is not expedient.
![image](../Images/N5_DTW_result.png)

Both waves are then transformed into the frequency range. This transformation is used to determine the pulse. 
![image](../Images/fft_plot.png)

To compile a 35s (1920x1080) video on the Pi4 took about 5min. Using the Pi5 cut the computational time in half quite exactly. Reducing the video quality also improved the computational time, while at the same time the results were as good as before regarding the DTW. 

The GUI is shown in the main folders README.
## Structure

```bash
Main_GUI_Pi/
│
│
├── Analysis_GUI/
│   └── HeartRateAnalyzer
│        ├── initUI
│        └── analyze
│
│
├── BVP_Analysis/
│   ├── get_frame_timestamps
│   ├── calculate_frame_rate
│   ├── visualize_BVPs
│   ├── PPG_data
│   ├── BVP_data
│   ├── peakfinder_rppg
│   ├── normalize
│   ├── plot_warping
│   ├── DTW_analysis
│   ├── analysis
│   └── Pipeline_analysis
│        └── run_on_video
│
│
├── Main_GUI/
│   └── HeartRateAnalyzer
│        ├── initUI
│        └── start_program
│
│
├── heartrate_monitor/
│   └── HeartRateMonitor
│        ├── run_sensor
│        ├── start_sensor
│        └── stop_sensor
│
│
├── hrcalc/
│   ├── calc_hr_and_spo2
│   ├── find_peaks
│   ├── find_peaks_above_min_height
│   └── Pipeline_analysis
│
│
└── max30102/
     └── MAX30102
          ├── shutdown
          ├── reset
          ├── setup
          ├── set_config
          ├── get_data_present
          ├── read_fifo
          └── read_sequential

```
Due to privacy reasons, no video data or PPG data utilized in this work is available.
Also make sure to raplace the 'path' in the main file with an actual folder where vidoes are stored!
