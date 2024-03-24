
# Multiprocessing
Acceleration for Holistic approach by multyprocessing frames parallel.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![pyVHR 2.0](https://img.shields.io/badge/pyVHR-2.0-blue.svg)](https://pypi.org/project/pyVHR/)
![GitHub last commit](https://img.shields.io/github/last-commit/morijx/Raspi_cardiac_wave)
![GitHub license](https://img.shields.io/github/license/morijx/Raspi_cardiac_wave)

## Summary
The introduction of a modified signal extraction method, alongside parallel processing of video frames within the Holistic approach, yielded a noticeable enhancement in computational efficiency on the MacBook platform. Used is the pyVHR-CPU package!

Looking again at the original function structure, we can see that the frames are passed through one after the other to produce the signal. At this point, however, it should also be possible to process several frames at the same time on different cores and extract the signals. The function was rewritten for this purpose.

When evaluating the computational performance of each component within the pyVHR algorithm, the following list presents the respective time required for computation.
| Part of the Model                    |  Time [s]  |
|--------------------------------------|------------|
| convexhul                            | 9.5367431640625e-07    |
| extract RGB signal with hol approach | 33.30496311187744       |
| windowing                            | 0.00010180473327636719 |
| RGB to BVP method                    | 0.009060859680175781    |
| **Total**                            | **33.31685709953308**   |

## Results

| Number of Parallel Processes | Computational Time [s] |
|------------------------------|------------------------|
| 1 (no multiprocessing)       | 33.30496311187744      |
| 3                            | 25.110912084579468     |
| 4                            | 26.02312707901001      |
| 5                            | 27.135556936264038     |
| 10                           | 39.9052369594574       |


As it is shown in the table a reduction is clearly visible. From no multiprocessing to 3 parallel processes a time reduction of 24.6 \% is possible for the generation of the RGB signal. The calculation times shown refer to a 35s video with 1920x1080 resolution calculated on a MacBook Pro M1. 


## Use

To use the code just start the Main_GUI.py. Keep in mind that the pyVHR-CPU package has to be installed correctly since it is needed in the process.



