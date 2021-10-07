# Jetson Stereo Depth

HW-accelerated Stereo Depth on Nvidia Jetson Platform

![demo gif](docs/output_disp_rgb_hstack.gif)


This repository contains:

- [x] HW-accelerated depth pipeline _(capture->rectify->SGBM->depth calc.)_ in python
- [x] Reference disparity pipeline with OpenCV CPU implementation
- [x] Calibration notebooks to determine a) intrinsic parameters and lens distortion coeficients of cameras b) rectification map between two sensors 
- [ ] Deployment code in C++
