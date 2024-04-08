import cv2
import mediapipe as mp
import numpy as np
from pyVHR.extraction.utils import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.extraction.sig_extraction_methods import *

import multiprocessing

def extract_skin(frame, ldmks):
    """
    Extract skin pixels from a frame using a predefined skin color range in HSV color space.

    This function converts the input frame to the HSV color space and creates a mask to isolate skin pixels based on
    predefined lower and upper bounds for skin color. It then applies the mask to the original frame to extract skin
    regions.

    Parameters:
        frame (numpy.ndarray): The input frame in BGR color space.
        ldmks (list): List of facial landmarks.

    Returns:
        numpy.ndarray, numpy.ndarray: The skin regions extracted from the frame and the binary mask used for extraction.
    """
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask to isolate skin pixels
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Apply the mask to the original frame
    skin_im = cv2.bitwise_and(frame, frame, mask=mask)

    # Optionally, you can perform additional processing or refinement on the skin_im

    return skin_im, mask

def compute_rgb_mean(cropped_skin_im, low_th, high_th):
    """
    Compute the mean RGB values of a cropped skin image.

    This function converts the input cropped skin image to float32 to prevent overflow during calculations.
    It then computes the mean RGB values and optionally applies any preprocessing or post-processing if needed.

    Parameters:
        cropped_skin_im (numpy.ndarray): The cropped skin image in BGR color space.
        low_th (float): Lower threshold for preprocessing.
        high_th (float): Upper threshold for preprocessing.

    Returns:
        numpy.ndarray: The mean RGB values of the cropped skin image.
    """
    # Convert the image to float32 to prevent overflow during calculations
    cropped_skin_im = cropped_skin_im.astype(np.float32)

    # Optionally, apply any preprocessing to the image if needed

    # Compute mean RGB values
    rgb_mean = np.mean(cropped_skin_im, axis=(0, 1))

    # Optionally, you can perform additional post-processing or filtering on the rgb_mean

    return rgb_mean

def process_frame(frame, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD):
    """
    Process a single frame to extract facial landmarks and compute skin color information.

    This function initializes FaceMesh and other necessary components to detect facial landmarks in the input frame.
    It then extracts facial landmarks and filters them based on visibility and presence thresholds.
    Next, it uses the extracted landmarks to extract skin regions from the frame and computes the mean RGB values
    of the cropped skin region. If no face or insufficient facial landmarks are detected, it returns None.

    Parameters:
        frame (numpy.ndarray): The input frame in BGR color space.
        PRESENCE_THRESHOLD (float): The presence threshold for facial landmarks.
        VISIBILITY_THRESHOLD (float): The visibility threshold for facial landmarks.

    Returns:
        numpy.ndarray or None: The mean RGB values of the cropped skin region if a face is detected and landmarks
        pass the thresholds, otherwise None.
    """
    # Initialize FaceMesh and other necessary components
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [l for l in face_landmarks.landmark]
        ldmks = np.zeros((468, 5), dtype=np.float32)
        for idx, landmark in enumerate(landmarks):
            if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                    or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                coords = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image.shape[1], image.shape[0])
                if coords:
                    ldmks[idx, 0] = coords[1]
                    ldmks[idx, 1] = coords[0]
        cropped_skin_im, full_skin_im = extract_skin(image, ldmks)
        return compute_rgb_mean(cropped_skin_im, SignalProcessingParams.RGB_LOW_TH, SignalProcessingParams.RGB_HIGH_TH)
    return None

def extract_holistic_parallel(videoFileName, max_processes=3, maxchilds = 75):
    """
    Extract holistic information from each frame of a video in parallel.

    This function reads the input video file and processes its frames in parallel using a multiprocessing Pool.
    Each frame is resized to a target width and height before processing. The function applies the `process_frame`
    function to each resized frame in parallel, limiting the number of processes and tasks per child process.
    Holistic information, such as mean skin color, is extracted from each frame. The function returns an array
    containing the extracted information.

    Parameters:
        videoFileName (str): The path to the input video file.
        max_processes (int): The maximum number of processes to use for parallel processing (default is 3).
        maxchilds (int): The maximum number of tasks each child process can complete before it is replaced (default is 75).

    Returns:
        numpy.ndarray: An array containing holistic information extracted from each frame of the video.
    """
    # Read video
    cap = cv2.VideoCapture(videoFileName)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_width = 640
    target_height = 480
    
    # Process frames in parallel with limited number of processes
    with multiprocessing.Pool(processes=max_processes, maxtasksperchild = maxchilds) as pool:
        results = []
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5
        for _ in range(num_frames):
            ret, frame = cap.read()
            resized_frame = cv2.resize(frame, (target_width, target_height))
            results.append(pool.apply_async(process_frame, (resized_frame, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD)))
        
        # Collect results
        sig = [result.get() for result in results if result.get() is not None]
    
    cap.release()
    
    return np.array(sig, dtype=np.float32)
