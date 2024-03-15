# Video Success


Using the notebooks in this folder, a classifier was trained to determine whether the video would give good results using the pyVHR pipeline, based on various features. 

## Features
- Brightness of the face: This measures the overall luminance or brightness level of the region
corresponding to the face within the video frame.
- Brightness of the whole frame: This represents the average brightness level across the entire video
frame, including all objects and regions.
- Average PSNR: (Peak Signal-to-Noise Ratio) measures the average quality of video reconstruction
between consecutive frames, indicating how closely the grey-level intensity of the current frame
matches the intensity of the previous frame, with higher values suggesting better fidelity.
- Mean feature displacements: This measures the average movement or displacement of certain
features, likely facial landmarks, across consecutive frames of the video.
- Face skin tone: This represents the average skin colour tone within the face region of the video
frame, typically represented in terms of hue, saturation, and value.


![image](Images/Feature_importance_all.png)

