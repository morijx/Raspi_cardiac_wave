import cv2
import mediapipe as mp
import numpy as np
from pyVHR.extraction.utils import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.extraction.sig_extraction_methods import *

import multiprocessing

def extract_skin(frame, ldmks):
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
    # Convert the image to float32 to prevent overflow during calculations
    cropped_skin_im = cropped_skin_im.astype(np.float32)

    # Optionally, apply any preprocessing to the image if needed

    # Compute mean RGB values
    rgb_mean = np.mean(cropped_skin_im, axis=(0, 1))

    # Optionally, you can perform additional post-processing or filtering on the rgb_mean

    return rgb_mean

def process_frame(frame, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD):
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

def extract_holistic_parallel(videoFileName, max_processes=5):
    # Read video
    cap = cv2.VideoCapture(videoFileName)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_width = 640
    target_height = 480
    
    # Process frames in parallel with limited number of processes
    with multiprocessing.Pool(processes=max_processes) as pool:
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
