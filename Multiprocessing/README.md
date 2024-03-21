
# Multiprocessing
Acceleration for Holistic approach by multyprocessing frames parallel.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![pyVHR 2.0](https://img.shields.io/badge/pyVHR-2.0-blue.svg)](https://pypi.org/project/pyVHR/)
![GitHub last commit](https://img.shields.io/github/last-commit/morijx/Raspi_cardiac_wave)
![GitHub license](https://img.shields.io/github/license/morijx/Raspi_cardiac_wave)

## Summary
The introduction of a modified signal extraction method, alongside parallel processing of video frames within the Holistic approach, yielded a noticeable enhancement in computational efficiency on the MacBook platform. Used is the pyVHR-CPU package!


## Results

| Number of Parallel Processes | Computational Time [s] |
|------------------------------|------------------------|
| 1 (no multiprocessing)      | 33.30496311187744      |
| 3                            | 25.110912084579468     |
| 4                            | 26.02312707901001      |
| 5                            | 27.135556936264038     |
| 10                           | 39.9052369594574       |


As it is shown in the table a reduction is clearly visible. From no multiprocessing to 3 parallel processes a time reduction of 24.6 \% is possible for the generation of the RGB signal. 






