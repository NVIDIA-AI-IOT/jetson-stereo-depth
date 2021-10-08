# Jetson Stereo Depth

HW-accelerated Stereo Depth on Nvidia Jetson Platform

![demo gif](docs/output_disp_rgb_hstack.gif)


This repository contains:

- [x] HW-accelerated depth pipeline _(capture->rectify->SGBM->depth calc.)_ in python
- [x] Reference disparity pipeline with OpenCV CPU implementation
- [x] Calibration notebooks to determine a) intrinsic parameters and lens distortion coeficients of cameras b) rectification map between two sensors 
- [ ] Deployment code in C++

## Pipeline 
![depth_pipeline](https://user-images.githubusercontent.com/26127866/136469605-e870fe16-9e91-4a8c-82f9-f126b3d5d64a.png)


## Hardware

3D printable holder for 2 × IMX219 camera module. Baseline distance of the pair is 100mm.
![stere_pair](https://user-images.githubusercontent.com/26127866/136617929-e856610c-76fe-4e02-a634-4d3fe37d1f12.png)
