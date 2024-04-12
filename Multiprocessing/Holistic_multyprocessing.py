import cv2
import mediapipe as mp
import numpy as np
from pyVHR.extraction.utils import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.extraction.sig_extraction_methods import *

import multiprocessing


def holistic_mean(im, RGB_LOW_TH, RGB_HIGH_TH):
    """
    This method computes the RGB-Mean Signal excluding 'im' pixels
    that are outside the RGB range [RGB_LOW_TH, RGB_HIGH_TH] (extremes are included).

    Args: 
        im (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        RGB_LOW_TH (numpy.int32): RGB low threshold value.
        RGB_HIGH_TH (numpy.int32): RGB high threshold value.
    
    Returns:
        RGB-Mean Signal as float32 ndarray with shape [1,3], where 1 is the single estimator,
        and 3 are r-mean, g-mean and b-mean.
    """
    mean = np.zeros((1, 3), dtype=np.float32)
    mean_r = np.float32(0.0)
    mean_g = np.float32(0.0)
    mean_b = np.float32(0.0)
    num_elems = np.float32(0.0)
    for x in prange(im.shape[0]):
        for y in prange(im.shape[1]):
            if not((im[x, y, 0] <= RGB_LOW_TH and im[x, y, 1] <= RGB_LOW_TH and im[x, y, 2] <= RGB_LOW_TH)
                    or (im[x, y, 0] >= RGB_HIGH_TH and im[x, y, 1] >= RGB_HIGH_TH and im[x, y, 2] >= RGB_HIGH_TH)):
                mean_r += im[x, y, 0]
                mean_g += im[x, y, 1]
                mean_b += im[x, y, 2]
                num_elems += 1.0
    if num_elems > 1.0:
        mean[0, 0] = mean_r / num_elems
        mean[0, 1] = mean_g / num_elems
        mean[0, 2] = mean_b / num_elems
    else:
        mean[0, 0] = mean_r
        mean[0, 1] = mean_g
        mean[0, 2] = mean_b 
    return mean

def extract_skin1(image, ldmks):
    """
    This method extract the skin from an image using Convex Hull segmentation.

    Args:
        image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
        ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].

    Returns:
        Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
    """

    from pyVHR.extraction.sig_processing import MagicLandmarks
    aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]        
    # face_mask convex hull 
    hull = ConvexHull(aviable_ldmks)
    verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
    img = Image.new('L', image.shape[:2], 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)
    mask = np.expand_dims(mask,axis=0).T

    # left eye convex hull
    """left_eye_ldmks = ldmks[MagicLandmarks.left_eye]
    aviable_ldmks = left_eye_ldmks[left_eye_ldmks[:,0] >= 0][:,:2]
    if len(aviable_ldmks) > 3:
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        left_eye_mask = np.array(img)
        left_eye_mask = np.expand_dims(left_eye_mask,axis=0).T
    else:
        left_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)"""

    # right eye convex hull
    """right_eye_ldmks = ldmks[MagicLandmarks.right_eye]
    aviable_ldmks = right_eye_ldmks[right_eye_ldmks[:,0] >= 0][:,:2]
    if len(aviable_ldmks) > 3:
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        right_eye_mask = np.array(img)
        right_eye_mask = np.expand_dims(right_eye_mask,axis=0).T
    else:
        right_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)"""

    # mounth convex hull
    """mounth_ldmks = ldmks[MagicLandmarks.mounth]
    aviable_ldmks = mounth_ldmks[mounth_ldmks[:,0] >= 0][:,:2]
    if len(aviable_ldmks) > 3:
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mounth_mask = np.array(img)
        mounth_mask = np.expand_dims(mounth_mask,axis=0).T
    else:
        mounth_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)"""

    # apply masks and crop 
    skin_image = image * mask #* (1-left_eye_mask) * (1-right_eye_mask) * (1-mounth_mask)

    rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)

    cropped_skin_im = skin_image
    if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
        cropped_skin_im = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]

    return cropped_skin_im, skin_image

def process_frame(frame, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD):
    """
    Process a single frame to extract skin color information.

    This function initializes the FaceMesh model and other necessary components to detect facial landmarks in the input frame.
    It then extracts skin color information from the detected facial landmarks by first converting the frame to RGB color space,
    detecting landmarks using FaceMesh, and extracting skin pixels using predefined thresholds for presence and visibility.
    The function returns the mean RGB values of the cropped skin region if a face is detected, otherwise returns None.

    Parameters:
        frame (numpy.ndarray): The input frame in BGR color space.
        PRESENCE_THRESHOLD (float): The presence threshold for facial landmarks.
        VISIBILITY_THRESHOLD (float): The visibility threshold for facial landmarks.

    Returns:
        numpy.ndarray or None: The mean RGB values of the cropped skin region if a face is detected, otherwise None.
    """
    sig = []
    
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
        cropped_skin_im, full_skin_im = extract_skin1(image, ldmks)# extract skin
        sig.append(holistic_mean(
                    cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
        return sig #compute_rgb_mean(cropped_skin_im, SignalProcessingParams.RGB_LOW_TH, SignalProcessingParams.RGB_HIGH_TH)
    return None

def extract_holistic_parallel(videoFileName, max_processes=5, maxchild = 90):
    """
    Extract holistic skin color information from a video in parallel.

    This function reads the input video file, processes its frames in parallel using a limited number of processes,
    and extracts holistic skin color information from each frame. It resizes each frame to a target width and height,
    then applies multiprocessing to process frames in parallel with the specified number of processes.
    The function returns an array containing the extracted skin color information for each frame.

    Parameters:
        videoFileName (str): The path to the input video file.
        max_processes (int): The maximum number of processes to use for parallel processing.

    Returns:
        numpy.ndarray: An array containing the extracted skin color information for each frame in the video.
    """
    cap = cv2.VideoCapture(videoFileName)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frames in parallel with limited number of processes
    with multiprocessing.Pool(processes=max_processes, maxtasksperchild= maxchild) as pool:
        results = []
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5
        for _ in range(num_frames):
            ret, frame = cap.read()
            resized_frame = frame 
            results.append(pool.apply_async(process_frame, (resized_frame, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD)))
        
        # Collect results
        sig = [result.get() for result in results if result.get() is not None]
    
    cap.release()
    
    return np.array(sig, dtype=np.float32)
