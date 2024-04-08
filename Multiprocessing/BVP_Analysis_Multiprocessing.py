import cv2
import pandas as pd
import numpy as np
import time as ttime
from scipy.signal import find_peaks, cheby2
from scipy.fft import fft, fftfreq
from scipy import signal
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import configparser
import ast
from numpy.lib.arraysetops import isin
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from importlib import import_module, util
from pyVHR.datasets.dataset import datasetFactory
from pyVHR.utils.errors import getErrors, printErrors, displayErrors, BVP_windowing
#from pyVHR.extraction.sig_processing import *
from pyVHR.extraction.sig_extraction_methods import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.BVP.BVP import *
from pyVHR.BPM.BPM import *
from pyVHR.BVP.methods import *
from pyVHR.BVP.filters import *
import time
from inspect import getmembers, isfunction
import os.path
from pyVHR.deepRPPG.mtts_can import *


from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from pyVHR.BVP.utils import jadeR

from Holistic import SignalProcessing_fast as SP
from Holistic import *

from Holistic_multyprocessing import extract_holistic_parallel as ehp


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def calculate_brightness(frame):
    """
    Calculate the average brightness of a frame.

    This function takes a frame (image) as input and calculates the average brightness across all color channels.
    
    Parameters:
        frame (numpy.ndarray): The input frame (image) represented as a numpy array.
    
    Returns:
        int: The calculated average brightness value, rounded to the nearest integer.
    """
    # Calculate average brightness across color channels
    brightness = int(frame.mean())
    return brightness

def detect_faces(frame):
    """
    Detect faces in a given frame using a cascade classifier.

    This function detects faces in the provided frame using a pre-trained cascade classifier.
    
    Parameters:
        frame (numpy.ndarray): The input frame (image) in which faces are to be detected.
    
    Returns:
        numpy.ndarray: A NumPy array containing information about the detected faces.
            Each row represents a detected face, with the format [x, y, w, h], where:
            - (x, y) are the coordinates of the top-left corner of the bounding box,
            - w is the width, and
            - h is the height of the bounding box.
    """
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    return faces

def main_brightness(video_path):
    """
    Calculate the mean brightness of faces in a video.

    This function reads a video file, detects faces in each frame, calculates the brightness of each face,
    and returns the mean brightness of all detected faces in the video.
    
    Parameters:
        video_path (str): The path to the input video file.
    
    Returns:
        float: The mean brightness of faces in the video.
            If no faces are detected, it returns "Error".
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    total_brightness = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Iterate through detected faces and calculate brightness
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            brightness = calculate_brightness(face_roi)
            total_brightness += brightness

        total_frames += 1

    cap.release()

    if total_frames > 0:
        mean_brightness = total_brightness / total_frames
        #print("Mean brightness of faces in the video:", mean_brightness)
        return mean_brightness
    else:
        print("No faces detected in the video.")
        return "Error"
    



def run_on_video( videoFileName, cuda=False, roi_method='convexhull', roi_approach='hol', method='cpu_POS', bpm_type='welch', pre_filt=False, post_filt=True, verb=False, win_size = 5):
    """ 
    Runs the pipeline on a specific video file.

    Args:
        videoFileName:
            - The path to the video file to analyse
        cuda:
            - True - Enable computations on GPU
            - False - Use CPU only
        roi_method:
            - 'convexhull' - Uses MediaPipe's lanmarks to compute the convex hull of the face and segment the skin
            - 'faceparsing' - Uses BiseNet to parse face components and segment the skin
        roi_approach:
            - 'hol' - Use the Holistic approach (one single ROI defined as the whole face skin region of the subject)
            - 'patches' - Use multiple patches as Regions of Interest
        method:
            - One of the rPPG methods defined in pyVHR
        bpm_type:
            - the method for computing the BPM estimate on a time window
        pre_filt:
            - True - Use Band pass filtering on the windowed RGB signal
            - False - No pre-filtering
        post_filt:
            - True - Use Band pass filtering on the estimated BVP signal
            - False - No post-filtering
        verb:
            - False - not verbose
            - True - show the main steps


    """


    Starttime = time.time()
    ldmks_list = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]
    assert os.path.isfile(videoFileName), "\nThe provided video file does not exists!"

    high_prio_forehead = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
    ldmks_list = high_prio_forehead
    
    sig_processing = SP()
    av_meths = getmembers(pyVHR.BVP.methods, isfunction)
    available_methods = [am[0] for am in av_meths]

    assert method in available_methods, "\nrPPG method not recognized!!"

    time1 = time.time()
    
    # set skin extractor
    if roi_method == 'convexhull':
        sig_processing.set_skin_extractor(
            SkinExtractionConvexHull())
    elif roi_method == 'faceparsing':
        sig_processing.set_skin_extractor(
            SkinExtractionFaceParsing())
    else:
        raise ValueError("Unknown 'roi_method'")
    
    time2 = time.time()
    
    assert roi_approach == 'patches' or roi_approach=='hol', "\nROI extraction approach not recognized!"
    
    # set patches
    if roi_approach == 'patches':
        #ldmks_list = ast.literal_eval(landmarks_list)
        #if len(ldmks_list) > 0:
        sig_processing.set_landmarks(ldmks_list)
        # set squares patches side dimension
        sig_processing.set_square_patches_side(28.0)
    
    # set sig-processing and skin-processing params
    SignalProcessingParams.RGB_LOW_TH = 75
    SignalProcessingParams.RGB_HIGH_TH = 230
    SkinProcessingParams.RGB_LOW_TH = 75
    SkinProcessingParams.RGB_HIGH_TH = 230

    if verb:
        print('\nProcessing Video: ' + videoFileName)
    fps = get_fps(videoFileName)
    sig_processing.set_total_frames(0)

    # -- ROI selection
    sig = []
    if roi_approach == 'hol':
        # SIG extraction with holistic
        sig = ehp(videoFileName)
    elif roi_approach == 'patches':
        # SIG extraction with patches
        sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')

    #np.savetxt('Multiprocess_.txt', sig)
    sig = sig.reshape((-1, 1, 3))

    time3 = time.time()
    # -- sig windowing
    windowed_sig, timesES = sig_windowing(sig, 6, 1, fps)

    time4 = time.time()

    # -- PRE FILTERING
    filtered_windowed_sig = windowed_sig

    # -- color threshold - applied only with patches
    if roi_approach == 'patches':
        filtered_windowed_sig = apply_filter(windowed_sig,
                                                rgb_filter_th,
                                                params={'RGB_LOW_TH':  75,
                                                        'RGB_HIGH_TH': 230})

    if pre_filt:
        module = import_module('pyVHR.BVP.filters')
        method_to_call = getattr(module, 'BPfilter')
        filtered_windowed_sig = apply_filter(filtered_windowed_sig, 
                            method_to_call, 
                            fps=fps, 
                            params={'minHz':0.65, 'maxHz':4.0, 'fps':'adaptive', 'order':6})
    if verb:
        print("\nBVP extraction with method: %s" % (method))

    # -- BVP Extraction
    module = import_module('methods_Luca')
    method_to_call = getattr(module, method)
    
    if 'cpu' in method:
        method_device = 'cpu'
    elif 'torch' in method:
        method_device = 'torch'
    elif 'cupy' in method:
        method_device = 'cuda'

    if 'POS' in method:
        pars = {'fps':'adaptive'}
    elif 'PCA' in method or 'ICA' in method:
        pars = {'component': 'all_comp'}
    else:
        pars = {}

    if method == "cpu_LGI_TSVD":
        pars = {'Stress':'True'}

    time5 = time.time()
    bvps = RGB_sig_to_BVP(filtered_windowed_sig, fps,
                            device_type=method_device, method=method_to_call, params=pars)
    
    time6 = time.time()
    # -- POST FILTERING
    if post_filt:
        module = import_module('pyVHR.BVP.filters')
        method_to_call = getattr(module, 'BPfilter')
    

        bvps = apply_filter(bvps, 
                            method_to_call, 
                            fps=fps, 
                            params={'minHz':0.65, 'maxHz':4.0, 'fps':'adaptive', 'order':6})

    if verb:
        print("\nBPM estimation with: %s" % (bpm_type))
    # -- BPM Estimation
    if bpm_type == 'welch':
        bpmES = BVP_to_BPM(bvps, fps, minHz=0.65, maxHz=4.0)
    elif bpm_type == 'psd_clustering':
        bpmES = BVP_to_BPM_PSD_clustering(bvps, fps, minHz=0.65, maxHz=4.0)
    else:
        raise ValueError("Unknown 'bpm_type'")

    time7 = time.time()
    # median BPM from multiple estimators BPM
    median_bpmES, mad_bpmES = multi_est_BPM_median(bpmES)

    if verb:
        print('\n...done!\n')
    time8 = time.time()

    delta1 = time2-time1
    delta2 = time3-time2
    delta3 = time4-time3
    delta4 = time6-time5

    print(f"faceparsing / convexhul: {delta1}")
    print(f"extract hol/patches: {delta2}")
    print(f"windowing: {delta3}")
    print(f"RGB to BVP: {delta4}")




    return timesES, median_bpmES, mad_bpmES, bvps

def cut_signal_at_peaks(signal, peaks, window_size):
    """
    Cut the signal around each peak.

    This function takes a signal and a list of peak indices, and cuts out a window of specified size 
    around each peak.

    Parameters:
        signal (numpy.ndarray): The input signal.
        peaks (list): A list of indices representing the peak positions in the signal.
        window_size (int): The size of the window to be cut around each peak. It must be an odd number.

    Returns:
        list: A list of cut signals. Each cut signal corresponds to a window around a peak.
    """
    cut_signals = []
    for peak in peaks:
        start = peak - window_size // 2
        end = peak + window_size // 2
        cut_signals.append(signal[start:end])
    return cut_signals

def plot_padded_array(padded_array, fs, plot_duration=1.0):
    """
    Plot segments of a padded array.

    This function plots segments of a 2D array with each row representing a segment, 
    typically used for visualizing biosignals like photoplethysmogram (PPG) waveforms.

    Parameters:
        padded_array (numpy.ndarray): The 2D array containing segments of the signal.
        fs (float): The sampling frequency of the signal.
        plot_duration (float, optional): The duration (in seconds) to plot. 
            Defaults to 1.0 second.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    num_segments = padded_array.shape[0]
    max_nonzero_index = np.max(np.where(padded_array != 0))
    max_index = min(int(plot_duration * fs), max_nonzero_index + 1)
    x_range = np.arange(max_index) / fs
    
    for i in range(num_segments):
        plt.plot(x_range, padded_array[i, :max_index], label=f"Segment {i+1}")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('BVP Wave Segments')
    #plt.legend()
    plt.grid(True)
    plt.show()


def create_padded_array_between_peaks(signal, peaks):
    """
    Create a padded array containing segments of a signal between consecutive peaks.

    This function extracts segments of the signal between consecutive peaks and pads them to the length 
    of the original signal. It returns a 2D array where each row represents a segment between peaks.

    Parameters:
        signal (numpy.ndarray): The input signal.
        peaks (list): A list of indices representing the peak positions in the signal.

    Returns:
        numpy.ndarray: A 2D array containing segments of the signal padded to the length of the original signal.
    """
    padded_array = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        cut_signal = signal[start:end]
        padded_array.append(np.pad(cut_signal, (0, len(signal) - len(cut_signal)), mode='edge'))
    return np.array(padded_array)



def get_frame_timestamps(file_path):
    """
    Get timestamps for each frame in a video file.

    This function opens a video file, calculates the timestamp for each frame based on its position,
    and returns a list of timestamps corresponding to each frame.

    Parameters:
        file_path (str): The path to the input video file.

    Returns:
        list: A list of timestamps for each frame in the video.
    """
    cap = cv2.VideoCapture(file_path)

    # Get frames per second and total frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the duration of each frame
    frame_duration = 1 / fps

    # Initialize list to store timestamps
    frame_timestamps = []

    # Iterate through each frame and calculate the timestamp
    for i in range(total_frames):
        timestamp = i * frame_duration
        frame_timestamps.append(timestamp)

    cap.release()
    return frame_timestamps

def calculate_frame_rate(file_path):
    """
    Calculate the frame rate of a video file.

    This function opens a video file, iterates through each frame to collect timestamps, 
    and calculates the frame rate based on the timestamps.

    Parameters:
        file_path (str): The path to the input video file.

    Returns:
        float: The frame rate of the video file in frames per second (FPS).
    """
    cap = cv2.VideoCapture(file_path)

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize list to store timestamps
    frame_timestamps = []

    # Iterate through each frame and collect timestamps
    for i in range(total_frames):
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds
        frame_timestamps.append(timestamp)
        _, _ = cap.read()  # Read next frame

    # Calculate frame rate
    frame_rate = (total_frames - 1) / (frame_timestamps[-1] - frame_timestamps[0])

    cap.release()
    return frame_rate


def visualize_BVPs(BVPs,timestamps, overlap_ratio=1-1/6):
    """
    This method creates a plotly plot for visualizing a range of windowed BVP signals. This method must be called
    inside a Jupyter notebook or a Colab notebook.

    Args:
        BVPs (list of float32 ndarray): windowed BPM signal as a list of length num_windows of float32 ndarray with shape [num_estimators, window_frames].
        overlap_ratio (float): the overlap ratio between consecutive windows.
    
    """


    window_duration = BVPs[0].shape[1]  # Assuming 30 samples per second

    #print(window_duration)
    #print(overlap_ratio)

    overlap_duration = int(window_duration * overlap_ratio)
    total_frames = (len(BVPs) - 1) * (window_duration - overlap_duration) + window_duration

    # Create a matrix to store all BVP signals with NaNs for gaps
    bvp_matrix = np.full((len(BVPs), total_frames), np.nan)

    for window, bvp in enumerate(BVPs):
        start_frame = window * (window_duration - overlap_duration)
        end_frame = start_frame + window_duration

        for i, e in enumerate(bvp):
            # Subtract the mean of the BVP window
            #e -= np.mean(e)
            #e = butter_lowpass_filter(e, cutoff_frequency=5, sampling_rate=30)
            #e = detrend(e, type='constant')

            # Ensure the size of the assigned slice matches the size of the BVP signal
            slice_size = min(end_frame - start_frame, len(e))
            
            # Store the BVP signal in the matrix
            bvp_matrix[window, start_frame:start_frame + slice_size] = e[:slice_size]

    # Plot each BVP signal from the matrix
    for window in range(len(BVPs)):
        x_values = np.arange(total_frames)
        y_values = bvp_matrix[window, :]
        name = "BVP_Window_" + str(window)
        #fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=name))

    # Compute the mean signal, disregarding NaN values
    mean_signal = np.nanmean(bvp_matrix, axis=0)

    timestamps = timestamps[:len(mean_signal)]
    mean_signal = mean_signal[:len(timestamps)]
    #print(len(mean_signal))
    #print(len(timestamps))

    return mean_signal, timestamps

def PPG_data(csv_file):
    """
    Preprocess PPG data from a CSV file.

    This function reads PPG data from a CSV file, preprocesses it, and returns the preprocessed data along with peak times and indices.

    Parameters:
        csv_file (str): The path to the CSV file containing PPG data.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray: 
            Preprocessed PPG data for two channels, peak times for both channels, peak indices for both channels, and the time axis.
    """
    dataname = csv_file 

    start_cutoff = 8
    max_hr = 200
    #number_of_segments = 10

    # Load the data
    df = pd.read_csv(dataname, header=None)

    l = df.size
    datalength = l
    # Sampling frequency
    fs = 25

    # Extract the data
    Y1 = df.iloc[0, start_cutoff:]
    Y2 = df.iloc[1, start_cutoff:]



    def cutof(data):
        # Sample data array

        # Define the threshold value
        threshold = -3000

        # Find the index where values are greater than or equal to the threshold
        index_above_threshold = next((i for i, x in enumerate(data) if x >= threshold), None)

        # Check if any values meet the threshold
        if index_above_threshold is not None:
            # Slice the array from the index where values meet the threshold
            data_cut_off = data[index_above_threshold:]
        else:
            # If no values meet the threshold, return an empty array or handle it as needed
            data_cut_off = np.array([])
        return data_cut_off

    Y1 = cutof(Y1-np.mean(Y1))
    Y2 = cutof(Y2-np.mean(Y2))



    # Create the time axis
    a = Y1.shape
    X = np.arange(a[0]) / fs
    X = X +(start_cutoff/fs)/2

    # Define the bandpass filter parameters (adjust these values as needed)
    lowcut = 0.2  # Lower cutoff frequency in Hz
    highcut = 5.0  # Upper cutoff frequency in Hz
    order = 3  # Filter order

    rs = 15

    Wn = [lowcut, highcut]

    # Design the bandpass filter
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    b, a = cheby2(order, rs, Wn, btype='band', analog=False, output='ba', fs=25)

    # Apply the bandpass filter to the data
    Y1_filtered = signal.filtfilt(b, a, Y1)
    Y2_filtered = signal.filtfilt(b, a, Y2)

    # Plot the filtered and unfiltered data


    min_peak_distance = 6  # Minimum distance between peaks (adjust as needed)

    Y01 = Y1_filtered
    Y02 = Y2_filtered

    #y1max = np.max(Y1_filtered)-np.max(Y1_filtered)/1.5
    #y2max = np.max(Y2_filtered)-np.max(Y2_filtered)/1.5

    #y1max= 0
    #y2max=0

    y1max = np.mean(Y1_filtered)
    y2max = np.mean(Y2_filtered)

    min_peak_prominence =120

    # Find peaks for both signals
    peaks1, _ = find_peaks(Y1_filtered, height=y1max, distance=min_peak_distance, prominence=min_peak_prominence)
    peaks2, _ = find_peaks(Y2_filtered, height=y2max, distance=min_peak_distance, prominence=min_peak_prominence)

    """# Update min_peak_distance based on the distances between consecutive peaks
    def update_min_peak_distance(peaks, last_peak_distance):
        if len(peaks) >= 2:
            new_min_peak_distance = max(min_peak_distance, last_peak_distance / 2)
            return new_min_peak_distance
        else:
            return min_peak_distance

    min_peak_distance = update_min_peak_distance(peaks1, np.diff(peaks1)[-1])
    min_peak_distance = update_min_peak_distance(peaks2, np.diff(peaks2)[-1])"""

    # Find peaks again with the updated min_peak_distance and filter based on the condition
    def filter_peaks(peaks, Y_filtered, y_max, min_distance_percentage=0.5):
        new_peaks = []

        for i in range(len(peaks) - 1):
            current_peak = peaks[i]
            next_peak = peaks[i + 1]
            peak_distance = next_peak - current_peak

            if i >=2:

                if peak_distance > min_distance_percentage * (np.abs(Y_filtered[current_peak]) + np.abs(Y_filtered[next_peak])) / 2:
                    new_peaks.append(current_peak)

        # Add the last peak if there are peaks
        if len(peaks) > 0:
            new_peaks.append(peaks[-1])

        return np.array(new_peaks)

    #peaks1 = filter_peaks(peaks1, Y1_filtered, y1max, min_distance_percentage=0.5)
    #peaks2 = filter_peaks(peaks2, Y2_filtered, y2max, min_distance_percentage=0.5)


    # Calculate time differences between consecutive peaks for both signals
    peak_times1 = np.array(peaks1) / fs
    peak_times2 = np.array(peaks2) / fs

    # Calculate BPM for each peak pair
    bpm_values1 = 60 / np.diff(peak_times1)
    bpm_values2 = 60 / np.diff(peak_times2)

    Y1_filtered = Y1_filtered[:len(Y2_filtered)]
    X = X[:len(Y2_filtered)]
    Y2_filtered = Y2_filtered[:len(Y1_filtered)]
    X = X[:len(Y1_filtered)]

    def assert_same_length(vector1, vector2):
        if len(vector1) != len(vector2):
            print("Error: Vectors have different lengths.")
            return False
        else:
            return True

    assert assert_same_length(Y1_filtered, Y2_filtered), "Vectors have different lengths."

    return Y1_filtered, Y2_filtered, peak_times1, peak_times2, peaks1, peaks2, X


def BVP_data(calculated_frame_rate, mean_signal, timestamps):
    """
    Preprocess BVP (Blood Volume Pulse) data.

    This function preprocesses BVP data, including filtering, and returns the preprocessed data along with timestamps.

    Parameters:
        calculated_frame_rate (float): The frame rate calculated from the video.
        mean_signal (numpy.ndarray): The mean signal extracted from the video frames.
        timestamps (numpy.ndarray): Timestamps corresponding to each frame in the video.

    Returns:
        numpy.ndarray, numpy.ndarray: Preprocessed BVP signal and corresponding timestamps.
    """
    max_hr = 200
    start_cutoff = 8


    # Sampling frequency
    fs = calculated_frame_rate

    # Extract the data
    Y1 = mean_signal[start_cutoff:]

    #print(len(Y1))

    video_length = timestamps[-1]


    """x_values_mean = np.linspace(0, video_length, len(mean_signal))
    x_values_mean = x_values_mean[start_cutoff:]
    x_values_mean = x_values_mean- x_values_mean[0]"""

    x_values_mean= timestamps[start_cutoff:]

    def cutof(data):
        # Sample data array

        # Define the threshold value
        threshold = -3000

        # Find the index where values are greater than or equal to the threshold
        index_above_threshold = next((i for i, x in enumerate(data) if x >= threshold), None)

        # Check if any values meet the threshold
        if index_above_threshold is not None:
            # Slice the array from the index where values meet the threshold
            data_cut_off = data[index_above_threshold:]
        else:
            # If no values meet the threshold, return an empty array or handle it as needed
            data_cut_off = np.array([])
        return data_cut_off

    Y1 = cutof(Y1-np.mean(Y1))


    # Define the bandpass filter parameters (adjust these values as needed)
    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 4.0  # Upper cutoff frequency in Hz
    order = 5  # Filter order

    rs = 15

    Wn = [lowcut, highcut]

    # Design the bandpass filter
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    b, a = cheby2(order, rs, Wn, btype='band', analog=False, output='ba', fs=30)

    # Apply the bandpass filter to the data
    Y1_filtered_rppg = signal.filtfilt(b, a, Y1)
    mean_singal_filtered = Y1_filtered_rppg

    return mean_singal_filtered, x_values_mean,

def peakfinder_rppg(Y1_filtered_rppg, min_peak_prominence):
    """
    Find peaks in the preprocessed RPPG signal.

    This function finds peaks in the preprocessed RPPG signal using the specified minimum peak prominence.

    Parameters:
        Y1_filtered_rppg (numpy.ndarray): The preprocessed RPPG signal.
        min_peak_prominence (float): The minimum prominence of peaks.

    Returns:
        numpy.ndarray: Indices of the detected peaks in the RPPG signal.
    """
    y1max = np.mean(Y1_filtered_rppg)


    #min_peak_prominence =0.2
    min_peak_distance = 6

    # Find peaks for both signals
    peaks1, _ = find_peaks(Y1_filtered_rppg, height=y1max, distance=min_peak_distance, prominence=min_peak_prominence)

    # Plot the filtered and unfiltered data

    

    return peaks1

def normalize(data):
    """
    Normalize data between -1 and 1 using Min-Max normalization.
    
    Parameters:
        data (array-like): Input data to be normalized.
        
    Returns:
        array-like: Normalized data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = -1 + 2 * (data - min_val) / (max_val - min_val)
    return normalized_data

def plot_warping(s1, s2, x1, x2, path, filename=None, fig=None, axs=None,
                 series_line_options=None, warping_line_options=None,
                 xlabel1=None, ylabel1=None, xlabel2=None, ylabel2=None, title=None):
    """
    Plot the optimal warping between two sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param x1: X-axis values for sequence s1.
    :param x2: X-axis values for sequence s2.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    :param fig: Matplotlib Figure object
    :param axs: Array of Matplotlib axes.Axes objects (length == 2)
    :param series_line_options: Dictionary of options to pass to matplotlib plot.
                                None will not pass any options.
    :param warping_line_options: Dictionary of options to pass to matplotlib ConnectionPatch.
                                 None will use {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    :param xlabel1: Label for x-axis of the first subplot.
    :param ylabel1: Label for y-axis of the first subplot.
    :param xlabel2: Label for x-axis of the second subplot.
    :param ylabel2: Label for y-axis of the second subplot.
    :param title: Title for the entire plot.
    :return: Figure, list[Axes]
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
    elif fig is None or axs is None:
        raise TypeError('The fig and axs arguments need to be both None or both instantiated.')

    if series_line_options is None:
        series_line_options = {}

    axs[0].plot(x1, s1, **series_line_options)
    axs[1].plot(x2, s2, **series_line_options)
    
    if xlabel1:
        axs[0].set_xlabel(xlabel1)
    if ylabel1:
        axs[0].set_ylabel(ylabel1)
    if xlabel2:
        axs[1].set_xlabel(xlabel2)
    if ylabel2:
        axs[1].set_ylabel(ylabel2)
    if title:
        fig.suptitle(title)

    plt.tight_layout()

    lines = []
    if warping_line_options is None:
        warping_line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}

    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        con = ConnectionPatch(xyA=[x1[r_c], s1[r_c]], coordsA=axs[0].transData,
                              xyB=[x2[c_c], s2[c_c]], coordsB=axs[1].transData, **warping_line_options)
        lines.append(con)

    for line in lines:
        fig.add_artist(line)

    if filename:
        plt.savefig(filename)
        #plt.close()
        fig, axs = None, None

    return fig, axs


def DTW_analysis(normalized_time_series_yrppg, normalized_time_series_yir,normalized_time_series_yred):
    """
    Perform Dynamic Time Warping (DTW) analysis between two time series.

    This function aligns two normalized time series using DTW and calculates the distance between them.

    Parameters:
        normalized_time_series_yrppg (numpy.ndarray): Normalized time series of the RPPG signal.
        normalized_time_series_yir (numpy.ndarray): Normalized time series of the IR signal.
        normalized_time_series_yred (numpy.ndarray): Normalized time series of the RED signal.

    Returns:
        list, float: The warping path between the two time series and the calculated distance.
    """
    high_freq_data_aligned= normalized_time_series_yrppg/2 +0.5
    low_freq_data_aligned = normalized_time_series_yir/2 +0.5
    path = dtw.warping_path(-low_freq_data_aligned,high_freq_data_aligned)
    path_norm = path
    dtwvis.plot_warping(-low_freq_data_aligned,high_freq_data_aligned, path, filename = 'test.png')

    dist = dtw.distance(-low_freq_data_aligned,high_freq_data_aligned)
    #print('Normalised Distance IR')
    #print(dist)


    low_freq_data_aligned = normalized_time_series_yred/2 +0.5
    high_freq_data_aligned= normalized_time_series_yrppg/2 +0.5

    path = dtw.warping_path(low_freq_data_aligned,high_freq_data_aligned)
    path_norm = path
    #dtwvis.plot_warping(low_freq_data_aligned,high_freq_data_aligned, path, filename = 'test.png')

    dist = dtw.distance(low_freq_data_aligned,high_freq_data_aligned)
    #print('Normalised Distance RED')
    #print(dist)
    return path_norm, dist


def analysis(video_file, csv_file, bvps):
    """
    Perform analysis on video and physiological data.

    This function conducts analysis on video and physiological data, including preprocessing, feature extraction,
    and Dynamic Time Warping (DTW) analysis.

    Parameters:
        video_file (str): The path to the video file.
        csv_file (str): The path to the CSV file containing physiological data.
        bvps (numpy.ndarray): Blood Volume Pulse (BVP) data.

    Returns:
        tuple: A tuple containing peaks detected in R, IR, and RPPG signals, aligned low-frequency and high-frequency signals,
        and related timestamps and DTW analysis results.
    """

    # Assuming 'method' is defined somewhere before calling the function
    window_range = (0, len(bvps))  # Adjust the range of windows to visualize
    #combined_bvp = visualize_BVPs(bvps, window_range)


    timestamps = get_frame_timestamps(video_file)
    calculated_frame_rate = calculate_frame_rate(video_file)

    mean_signal, timestamps = visualize_BVPs(bvps,timestamps,overlap_ratio=(1-(1/6)))
    timestamps = timestamps[:len(mean_signal)]

    Y1_filtered, Y2_filtered, peak_times1, peak_times2, peaks1, peaks2, PPG_time = PPG_data(csv_file)

    mean_singal_filtered, BVP_time = BVP_data(calculated_frame_rate, mean_signal, timestamps)

    normalized_time_series_yred = normalize(Y2_filtered)
    normalized_time_series_yrppg = normalize(mean_singal_filtered)
    normalized_time_series_yir = normalize(Y1_filtered)

    peaks_rppg = peakfinder_rppg(normalized_time_series_yrppg, min_peak_prominence = 0.35)
    peaks_r = peakfinder_rppg(normalized_time_series_yred , min_peak_prominence = 0.2)
    peaks_ir = peakfinder_rppg(normalized_time_series_yir, min_peak_prominence = 0.1)

    #print(len(peaks_rppg))
    #print(len(peaks_r))
    #print(len(peaks_ir))

    path_norm, dist = DTW_analysis(normalized_time_series_yrppg, -normalized_time_series_yir,-normalized_time_series_yred)

    low_freq_data_aligned = -normalized_time_series_yred/2 +0.5
    high_freq_data_aligned= normalized_time_series_yrppg/2 +0.5

    x1 = PPG_time
    y1 = low_freq_data_aligned

    #print(len(BVP_time))
    #print(len(high_freq_data_aligned))

    x2 = BVP_time
    y2 = high_freq_data_aligned#[:len(BVP_time)]

    

    return peaks_r, peaks_ir, peaks_rppg, y1, y2, x1, x2, path_norm, dist





