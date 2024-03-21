import cv2
import mediapipe as mp
import numpy as np
from pyVHR.extraction.utils import *
from pyVHR.extraction.skin_extraction_methods import *
from pyVHR.extraction.sig_extraction_methods import *

import multiprocessing

"""
This module defines classes or methods used for Signal extraction and processing.
"""


class SignalProcessing_fast():
    """
        This class performs offline signal extraction with different methods:

        - holistic.

        - squared / rectangular patches.
    """

    def __init__(self):
        # Common parameters #
        self.tot_frames = None
        self.visualize_skin_collection = []
        self.skin_extractor = SkinExtractionConvexHull()
        # Patches parameters #
        high_prio_ldmk_id, mid_prio_ldmk_id = get_magic_landmarks()
        self.ldmks = high_prio_ldmk_id + mid_prio_ldmk_id
        self.square = None
        self.rects = None
        self.visualize_skin = False
        self.visualize_landmarks = False
        self.visualize_landmarks_number = False
        self.visualize_patch = False
        self.font_size = 0.3
        self.font_color = (255, 0, 0, 255)
        self.visualize_skin_collection = []
        self.visualize_landmarks_collection = []

    def set_total_frames(self, n):
        """
        Set the total frames to be processed; if you want to process all the possible frames use n = 0.
        
        Args:  
            n (int): number of frames to be processed.
            
        """
        if n < 0:
            print("[ERROR] n must be a positive number!")
        self.tot_frames = int(n)

    def set_skin_extractor(self, extractor):
        """
        Set the skin extractor that will be used for skin extraction.
        
        Args:  
            extractor: instance of a skin_extraction class (see :py:mod:`pyVHR.extraction.skin_extraction_methods`).
            
        """
        self.skin_extractor = extractor

    def set_visualize_skin_and_landmarks(self, visualize_skin=False, visualize_landmarks=False, visualize_landmarks_number=False, visualize_patch=False):
        """
        Set visualization parameters. You can retrieve visualization output with the 
        methods :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.get_visualize_skin` 
        and :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.get_visualize_patches`

        Args:  
            visualize_skin (bool): The skin and the patches will be visualized.
            visualize_landmarks (bool): The landmarks (centers of patches) will be visualized.
            visualize_landmarks_number (bool): The landmarks number will be visualized.
            visualize_patch (bool): The patches outline will be visualized.
        
        """
        self.visualize_skin = visualize_skin
        self.visualize_landmarks = visualize_landmarks
        self.visualize_landmarks_number = visualize_landmarks_number
        self.visualize_patch = visualize_patch

    def get_visualize_skin(self):
        """
        Get the skin images produced by the last processing. Remember to 
        set :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.set_visualize_skin_and_landmarks`
        correctly.
        
        Returns:
            list of ndarray: list of cv2 images; each image is a ndarray with shape [rows, columns, rgb_channels].
        """
        return self.visualize_skin_collection

    def get_visualize_patches(self):
        """
        Get the 'skin+patches' images produced by the last processing. Remember to 
        set :py:meth:`pyVHR.extraction.sig_processing.SignalProcessing.set_visualize_skin_and_landmarks`
        correctly
        
        Returns:
            list of ndarray: list of cv2 images; each image is a ndarray with shape [rows, columns, rgb_channels].
        """
        return self.visualize_landmarks_collection

    def extract_raw(self, videoFileName):
        """
        Extracts raw frames from video.

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            ndarray: raw frames with shape [num_frames, height, width, rgb_channels].
        """

        frames = []
        for frame in extract_frames_yield(videoFileName):
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # convert to RGB

        return np.array(frames)

    ### HOLISTIC METHODS ###

    def extract_raw_holistic(self, videoFileName):
        """
        Locates the skin pixels in each frame. This method is intended for rPPG methods that use raw video signal.

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: raw signal as float32 ndarray with shape [num_frames, rows, columns, rgb_channels].
        """

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                ### sig computing ###
                sig.append(full_skin_im)
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return sig

    def extract_holistic(self, videoFileName):
        """
        This method computes the RGB-mean signal using the whole skin (holistic);

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, 1, rgb_channels]. The second dimension is 1 because
            the whole skin is considered as one estimators.
        """
        self.visualize_skin_collection = []

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                ### sig computing ###
                sig.append(holistic_mean(
                    cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return sig

    def extract_frames(self, video_file, num_frames=None):
        cap = cv2.VideoCapture(video_file)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if num_frames is not None and len(frames) >= num_frames:
                break
        cap.release()
        return frames    

    def extract_holistic_fast(self, videoFileName):
        """
        This method computes the RGB-mean signal using the whole skin (holistic);

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, 1, rgb_channels]. The second dimension is 1 because
            the whole skin is considered as one estimators.
        """
        self.visualize_skin_collection = []

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        # Resize image once before the loop
        target_width = 640  # Example target width
        target_height = 480  # Example target height

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            
            # Resize the first frame before the loop
            first_frame = self.extract_frames(videoFileName, num_frames=1)[0]
            resized_first_frame = cv2.resize(first_frame, (target_width, target_height))
            
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                ### sig computing ###
                sig.append(holistic_mean(
                    cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return sig


    def extract_holistic_faster(self, videoFileName):
        """
        This method computes the RGB-mean signal using the whole skin (holistic);

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, 1, rgb_channels]. The second dimension is 1 because
            the whole skin is considered as one estimators.
        """
        self.visualize_skin_collection = []

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        # Resize image once before the loop
        target_width = 640  # Example target width
        target_height = 480  # Example target height

        # Move FaceMesh initialization outside the loop
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            first_frame = self.extract_frames(videoFileName, num_frames=1)[0]
            resized_first_frame = cv2.resize(first_frame, (target_width, target_height))
            for frame in extract_frames_yield(videoFileName):
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx, landmark in enumerate(landmarks):
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                # Move skin extraction outside the if-else block for efficiency
                cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, ldmks)
                if self.visualize_skin:
                    self.visualize_skin_collection.append(full_skin_im)
                sig.append(holistic_mean(
                    cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return sig

    def extract_holistic_optimized(self, videoFileName):
        self.visualize_skin_collection = []

        skin_ex = self.skin_extractor
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        target_width = 640
        target_height = 480

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            # Read and resize the first frame
            first_frame = self.extract_frames(videoFileName, num_frames=1)[0]
            resized_first_frame = cv2.resize(first_frame, (target_width, target_height))
            
            # Iterate through frames
            for frame in extract_frames_yield(videoFileName):
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                
                # Process frame using FaceMesh
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    # Extract landmark coordinates
                    ldmks = np.zeros((468, 5), dtype=np.float32)
                    for idx, landmark in enumerate(landmarks):
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, image.shape[1], image.shape[0])
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]

                    # Extract skin
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(image, ldmks)
                    if self.visualize_skin:
                        self.visualize_skin_collection.append(full_skin_im)
                    # Compute RGB-mean signal
                    sig.append(holistic_mean(
                        cropped_skin_im, np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH)))
                    
                # Check if total frames limit is reached
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break

        # Convert to numpy array
        sig = np.array(sig, dtype=np.float32)
        return sig

  


    ### PATCHES METHODS ###




    def set_landmarks(self, landmarks_list):
        """
        Set the patches centers (landmarks) that will be used for signal processing. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            landmarks_list (list): list of positive integers between 0 and 467 that identify patches centers (landmarks).
        """
        if not isinstance(landmarks_list, list):
            print("[ERROR] landmarks_set must be a list!")
            return
        self.ldmks = landmarks_list

    def set_square_patches_side(self, square_side):
        """
        Sets the dimension of the square patches that will be used for signal processing. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            square_side (float): positive float that defines the length of the square patches.
        """
        if not isinstance(square_side, float) or square_side <= 0.0:
            print("[ERROR] square_side must be a positive float!")
            return
        self.square = square_side

    def set_rect_patches_sides(self, rects_dim):
        """
        Sets the dimension of each rectangular patch. There are 468 facial points you can
        choose; for visualizing their identification number please use :py:meth:`pyVHR.plot.visualize.visualize_landmarks_list`.

        Args:
            rects_dim (float32 ndarray): positive float32 np.ndarray of shape [num_landmarks, 2]. If the list of used landmarks is [1,2,3] 
                and rects_dim is [[10,20],[12,13],[40,40]] then the landmark number 2 will have a rectangular patch of xy-dimension 12x13.
        """
        if type(rects_dim) != type(np.array([])):
            print("[ERROR] rects_dim must be an np.ndarray!")
            return
        if rects_dim.shape[0] != len(self.ldmks) and rects_dim.shape[1] != 2:
            print("[ERROR] incorrect rects_dim shape!")
            return
        self.rects = rects_dim

    def extract_patches(self, videoFileName, region_type, sig_extraction_method):
        """
        This method computes the RGB-mean signal using specific skin regions (patches).

        Args:
            videoFileName (str): video file name or path.
            region_type (str): patches types can be  "squares" or "rects".
            sig_extraction_method (str): RGB signal can be computed with "mean" or "median". We recommend to use mean.

        Returns: 
            float32 ndarray: RGB signal as ndarray with shape [num_frames, num_patches, rgb_channels].
        """
        if self.square is None and self.rects is None:
            print(
                "[ERROR] Use set_landmarks_squares or set_landmarkds_rects before calling this function!")
            return None
        if region_type != "squares" and region_type != "rects":
            print("[ERROR] Invalid landmarks region type!")
            return None
        if sig_extraction_method != "mean" and sig_extraction_method != "median":
            print("[ERROR] Invalid signal extraction method!")
            return None

        ldmks_regions = None
        if region_type == "squares":
            ldmks_regions = np.float32(self.square)
        elif region_type == "rects":
            ldmks_regions = np.float32(self.rects)

        sig_ext_met = None
        if sig_extraction_method == "mean":
            if region_type == "squares":
                sig_ext_met = landmarks_mean
            elif region_type == "rects":
                sig_ext_met = landmarks_mean_custom_rect
        elif sig_extraction_method == "median":
            if region_type == "squares":
                sig_ext_met = landmarks_median
            elif region_type == "rects":
                sig_ext_met = landmarks_median_custom_rect

        self.visualize_skin_collection = []
        self.visualize_landmarks_collection = []

        skin_ex = self.skin_extractor

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        PRESENCE_THRESHOLD = 0.5
        VISIBILITY_THRESHOLD = 0.5

        sig = []
        processed_frames_count = 0

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            for frame in extract_frames_yield(videoFileName):
                # convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames_count += 1
                width = image.shape[1]
                height = image.shape[0]
                # [landmarks, info], with info->x_center ,y_center, r, g, b
                ldmks = np.zeros((468, 5), dtype=np.float32)
                ldmks[:, 0] = -1.0
                ldmks[:, 1] = -1.0
                magic_ldmks = []
                ### face landmarks ###
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [l for l in face_landmarks.landmark]
                    for idx in range(len(landmarks)):
                        landmark = landmarks[idx]
                        if not ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD)
                                or (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                            coords = mp_drawing._normalized_to_pixel_coordinates(
                                landmark.x, landmark.y, width, height)
                            if coords:
                                ldmks[idx, 0] = coords[1]
                                ldmks[idx, 1] = coords[0]
                    ### skin extraction ###
                    cropped_skin_im, full_skin_im = skin_ex.extract_skin(
                        image, ldmks)
                else:
                    cropped_skin_im = np.zeros_like(image)
                    full_skin_im = np.zeros_like(image)
                ### sig computing ###
                for idx in self.ldmks:
                    magic_ldmks.append(ldmks[idx])
                magic_ldmks = np.array(magic_ldmks, dtype=np.float32)
                temp = sig_ext_met(magic_ldmks, full_skin_im, ldmks_regions,
                                   np.int32(SignalProcessingParams.RGB_LOW_TH), np.int32(SignalProcessingParams.RGB_HIGH_TH))
                sig.append(temp)
                # visualize patches and skin
                if self.visualize_skin == True:
                    self.visualize_skin_collection.append(full_skin_im)
                if self.visualize_landmarks == True:
                    annotated_image = full_skin_im.copy()
                    color = np.array([self.font_color[0],
                                      self.font_color[1], self.font_color[2]], dtype=np.uint8)
                    for idx in self.ldmks:
                        cv2.circle(
                            annotated_image, (int(ldmks[idx, 1]), int(ldmks[idx, 0])), radius=0, color=self.font_color, thickness=-1)
                        if self.visualize_landmarks_number == True:
                            cv2.putText(annotated_image, str(idx),
                                        (int(ldmks[idx, 1]), int(ldmks[idx, 0])), cv2.FONT_HERSHEY_SIMPLEX, self.font_size,  self.font_color,  1)
                    if self.visualize_patch == True:
                        if region_type == "squares":
                            sides = np.array([self.square] * len(magic_ldmks))
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), sides, sides, color)
                        elif region_type == "rects":
                            annotated_image = draw_rects(
                                annotated_image, np.array(magic_ldmks[:, 1]), np.array(magic_ldmks[:, 0]), np.array(self.rects[:, 0]), np.array(self.rects[:, 1]), color)
                    self.visualize_landmarks_collection.append(
                        annotated_image)
                ### loop break ###
                if self.tot_frames is not None and self.tot_frames > 0 and processed_frames_count >= self.tot_frames:
                    break
        sig = np.array(sig, dtype=np.float32)
        return np.copy(sig[:, :, 2:])
    



from numba import prange, njit
import numpy as np
import cv2

def get_magic_landmarks():
    """ returns high_priority and mid_priority list of landmarks identification number """
    return [*MagicLandmarks.forehead_center, *MagicLandmarks.cheek_left_bottom, *MagicLandmarks.cheek_right_bottom], [*MagicLandmarks.forehoead_right, *MagicLandmarks.forehead_left, *MagicLandmarks.cheek_left_top, *MagicLandmarks.cheek_right_top]


class MagicLandmarks():
    """
    This class contains usefull lists of landmarks identification numbers.
    """
    high_prio_forehead = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
    high_prio_nose = [3, 4, 5, 6, 45, 51, 115, 122, 131, 134, 142, 174, 195, 196, 197, 198,
                      209, 217, 220, 236, 248, 275, 277, 281, 360, 363, 399, 419, 420, 429, 437, 440, 456]
    high_prio_left_cheek = [36, 47, 50, 100, 101, 116, 117,
                            118, 119, 123, 126, 147, 187, 203, 205, 206, 207, 216]
    high_prio_right_cheek = [266, 280, 329, 330, 346, 347,
                             347, 348, 355, 371, 411, 423, 425, 426, 427, 436]

    mid_prio_forehead = [8, 9, 21, 68, 103, 251,
                         284, 297, 298, 301, 332, 333, 372, 383]
    mid_prio_nose = [1, 44, 49, 114, 120, 121, 128, 168, 188, 351, 358, 412]
    mid_prio_left_cheek = [34, 111, 137, 156, 177, 192, 213, 227, 234]
    mid_prio_right_cheek = [340, 345, 352, 361, 454]
    mid_prio_chin = [135, 138, 169, 170, 199, 208, 210, 211,
                     214, 262, 288, 416, 428, 430, 431, 432, 433, 434]
    mid_prio_mouth = [92, 164, 165, 167, 186, 212, 322, 391, 393, 410]
    # more specific areas
    forehead_left = [21, 71, 68, 54, 103, 104, 63, 70,
                     53, 52, 65, 107, 66, 108, 69, 67, 109, 105]
    forehead_center = [10, 151, 9, 8, 107, 336, 285, 55, 8]
    forehoead_right = [338, 337, 336, 296, 285, 295, 282,
                       334, 293, 301, 251, 298, 333, 299, 297, 332, 284]
    eye_right = [283, 300, 368, 353, 264, 372, 454, 340, 448,
                 450, 452, 464, 417, 441, 444, 282, 276, 446, 368]
    eye_left = [127, 234, 34, 139, 70, 53, 124,
                35, 111, 228, 230, 121, 244, 189, 222, 143]
    nose = [193, 417, 168, 188, 6, 412, 197, 174, 399, 456,
            195, 236, 131, 51, 281, 360, 440, 4, 220, 219, 305]
    mounth_up = [186, 92, 167, 393, 322, 410, 287, 39, 269, 61, 164]
    mounth_down = [43, 106, 83, 18, 406, 335, 273, 424, 313, 194, 204]
    chin = [204, 170, 140, 194, 201, 171, 175,
            200, 418, 396, 369, 421, 431, 379, 424]
    cheek_left_bottom = [215, 138, 135, 210, 212, 57, 216, 207, 192]
    cheek_right_bottom = [435, 427, 416, 364,
                          394, 422, 287, 410, 434, 436]
    cheek_left_top = [116, 111, 117, 118, 119, 100, 47, 126, 101, 123,
                      137, 177, 50, 36, 209, 129, 205, 147, 177, 215, 187, 207, 206, 203]
    cheek_right_top = [349, 348, 347, 346, 345, 447, 323,
                       280, 352, 330, 371, 358, 423, 426, 425, 427, 411, 376]
    # dense zones used for convex hull masks
    left_eye = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mounth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]
    # equispaced facial points - mouth and eyes are excluded.
    equispaced_facial_points = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, \
             58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, \
             118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, \
             210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, \
             284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, \
             346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]

@njit(parallel=True)
def draw_rects(image, xcenters, ycenters, xsides, ysides, color):
    """
    This method is used to draw N rectangles on a image.
    """
    for idx in prange(len(xcenters)):
        leftx = int(xcenters[idx] - xsides[idx]/2)
        rightx = int(xcenters[idx] + xsides[idx]/2)
        topy = int(ycenters[idx] - ysides[idx]/2)
        bottomy = int(ycenters[idx] + ysides[idx]/2)
        for x in prange(leftx, rightx):
            if topy >= 0 and x >= 0 and x < image.shape[1]:
                image[topy, x, 0] = color[0]
                image[topy, x, 1] = color[1]
                image[topy, x, 2] = color[2]
            if bottomy < image.shape[0] and x >= 0 and x < image.shape[1]:
                image[bottomy, x, 0] = color[0]
                image[bottomy, x, 1] = color[1]
                image[bottomy, x, 2] = color[2]
        for y in prange(topy, bottomy):
            if leftx >= 0 and y >= 0 and y < image.shape[0]:
                image[y, leftx, 0] = color[0]
                image[y, leftx, 1] = color[1]
                image[y, leftx, 2] = color[2]
            if rightx < image.shape[1] and y >= 0 and y < image.shape[0]:
                image[y, rightx, 0] = color[0]
                image[y, rightx, 1] = color[1]
                image[y, rightx, 2] = color[2]
    return image

def sig_windowing(sig, wsize, stride, fps):
    """
    This method is used to divide a RGB signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray with shape [num_frames, num_estimators, rgb_channels].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        windowed signal as a list of length num_windows of float32 ndarray with shape [num_estimators, rgb_channels, window_frames],
        and a 1D ndarray of times in seconds,where each one is the center of a window.
    """
    N = sig.shape[0]
    block_idx, timesES = sliding_straded_win_offline(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        wind_signal = np.swapaxes(wind_signal, 0, 1)
        wind_signal = np.swapaxes(wind_signal, 1, 2)
        block_signals.append(wind_signal)
    return block_signals, timesES

def raw_windowing(raw_signal, wsize, stride, fps):
    """
    This method is used to divide a Raw signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray of images with shape [num_frames, rows, columns, rgb_channels].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        windowed signal as a list of length num_windows of float32 ndarray with shape [num_frames, rows, columns, rgb_channels],
        and a 1D ndarray of times in seconds,where each one is the center of a window.
    """
    N = raw_signal.shape[0]
    block_idx, timesES = sliding_straded_win_offline(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(raw_signal[st_frame: end_frame+1])
        block_signals.append(wind_signal)
    return block_signals, timesES

def sliding_straded_win_offline(N, wsize, stride, fps):
    """
    This method is used to compute all the info for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)

def get_fps(videoFileName):
    """
    This method returns the fps of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps

def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()