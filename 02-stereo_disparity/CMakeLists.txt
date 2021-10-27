cmake_minimum_required(VERSION 3.5)

project(test_vpi_streams)

find_package(vpi 1.1 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} vpi opencv_imgproc opencv_core)

target_link_libraries(${PROJECT_NAME} vpi opencv_imgproc opencv_core opencv_videoio opencv_highgui opencv_calib3d)