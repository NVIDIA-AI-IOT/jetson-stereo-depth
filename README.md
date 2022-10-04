# Jetson Stereo Depth

Build your own HW-accelerated stereo camera from scratch with Nvidia Jetson.

Demo:
![demo gif](docs/output_disp_rgb_hstack.gif)

The hardware:
![IMG_4761](https://user-images.githubusercontent.com/26127866/145352427-23b812aa-ef7f-4419-975a-5bb9d2ceae41.jpeg)


<!-- https://user-images.githubusercontent.com/26127866/146460871-049c93b7-6757-4c77-8cde-2b619fa015ce.mp4 -->


This repository contains:

1. Stereo Pair intrinsic calibration
<img width="1183" alt="image" src="https://user-images.githubusercontent.com/26127866/193731578-40897c70-d1c0-4890-af8e-02e36b93c061.png">
2. Lens distortion correction
<img width="1004" alt="image" src="https://user-images.githubusercontent.com/26127866/193731737-c2241136-488f-4a77-a3a4-cfe6973ee554.png">
3. VPI HW-accelerated `remap()`
<img width="1105" alt="image" src="https://user-images.githubusercontent.com/26127866/193731814-ff3d1252-8945-443c-8a10-2ceeb8275d60.png">
4. Depth calculation
![demo gif](docs/output_disp_rgb_hstack.gif)


- [x] HW-accelerated depth pipeline _(capture->rectify->SGBM->depth calc.)_ in python
- [x] Reference disparity pipeline with OpenCV CPU implementation
- [x] Calibration notebooks to determine a) intrinsic parameters and lens distortion coeficients of cameras b) rectification map between two sensors 
- [x] Deployment code in C++
- [ ] Assembly instructions for the hardware

## Getting started

1. Install dependencies
```shell
sudo apt install libnvvpi1 python3-vpi1 vpi1-dev
```

2. Calibrate cameras (see notebooks in [calib/](calib/01_intrinsics_lens_dist.ipynb) folder).

3. Try python implementation
```shell
cd depth_pipeline_python
python3 depth_vpi.py 
```

4. Try C++ implementation
```shell
cd depth_pipeline_cpp
mkdir build && cd build
cmake .. && make -j$(nproc)
./depth_pipeline
```

## Pipeline 
![depth_pipeline](https://user-images.githubusercontent.com/26127866/136469605-e870fe16-9e91-4a8c-82f9-f126b3d5d64a.png)

## Hardware

3D printable holder for 2 × IMX219 camera module is available [here](https://github.com/NVIDIA-AI-IOT/jetson-stereo-depth/blob/master/stl/stereo_front.stl). Baseline distance of the pair is 80mm.


