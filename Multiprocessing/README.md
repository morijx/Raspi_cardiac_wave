
# Multiprocessing
Acceleration for Holistic approach by multyprocessing frames parallel.

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pyVHR 2.0](https://img.shields.io/badge/pyVHR-2.0-blue.svg)](https://pypi.org/project/pyVHR/)
![GitHub last commit](https://img.shields.io/github/last-commit/morijx/Raspi_cardiac_wave)
![GitHub license](https://img.shields.io/github/license/morijx/Raspi_cardiac_wave)
![Multiproccessing YES](https://img.shields.io/badge/Multiprocessing-YES-green.svg)

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

The table now shows the extraction time from the RGB signal for different number of parrallel processes on the MacBook.

| Number of Parallel Processes | Computational Time [s] |
|------------------------------|------------------------|
| 1 (no multiprocessing)       | 33.30496311187744      |
| 3                            | 25.110912084579468     |
| 4                            | 26.02312707901001      |
| 5                            | 27.135556936264038     |
| 10                           | 39.9052369594574       |


As it is shown in the table a reduction is clearly visible. From no multiprocessing to 3 parallel processes a time reduction of 24.6 \% is possible for the generation of the RGB signal. The calculation times shown refer to a 35s video with 1920x1080 resolution calculated on a MacBook Pro M1. 

## Rasperry Pi

To enable multiprocessing on the Raspberry Pi, an additional step had to be added. This includes the limitation of the child processes for the individual parallel processes. The limitation prevents more RAM being required than is available. Please note that the Raspebrry Pi used in this work has 8GB of RAM. It will probably also work with less ram, but the child functions must be further reduced. 

Replace the Holostic_multiprocessing with the Holostic_multiprocessing_Pi or add the needed lines of code. If your computer is struggeling with the task it might also help to limit the amount of child processes. 

| Part of the Model                | No parallel processing [s]    | 3 Parallel max 20 child processes [s] |
|---------------------------------|--------------------------------|----------------------------------------|
| convexhul                       | 5.7220459e-6                   | 5.483627e-6                            |
| extract RGB with hol approach   | 198.03354                      | 111.13346                              |
| windowing                       | 0.000963                       | 0.000952                               |
| RGB to BVP method               | 0.060685                       | 0.024593                               |
| **Total**                       | **198.15987**                  | **111.192**                            |

The reduction in calculation time is very impressive and can probably be reduced even further with other parameters. Some are shown in the following Table.

| Test | Time RGB extraction [s] | Parallel processes | max child processes |
|------|--------------------------|--------------------|---------------------|
| 1    | 111.13346                | 3                  | 20                  |
| 2    | 101.821795               | 3                  | 50                  |
| 3    | 100.39366                | 4                  | 50                  |
| 4    | 98.26871                 | 3                  | 75                  |
| 5    | 132.8003                 | 3                  | 100                 |

To perform the test, the swap memory was expanded to the maximum of 2GB. However, as soon as swap was used to perform the calculation, the calculation time increased. This can be seen in example 5. It is therefore recommended to select the parameters in such a way that the maximum is utilised without having to access the swap. 

## Use

To use the code just start the Main_GUI.py. Keep in mind that the pyVHR-CPU package has to be installed correctly since it is needed in the process.
Also make sure to raplace the 'path' with an actual folder where vidoes are stored!
The Holstic file just contains functions that are also available in the pyVHR package. It was mainly used in the process for understanding and printing different values.

