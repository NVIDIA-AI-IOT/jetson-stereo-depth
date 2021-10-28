/*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/contrib/contrib.hpp> // for colormap
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/highgui/highgui.hpp> // imshow()

#include <opencv2/opencv.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <cstring> // for memset
#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

// #define CHECK_STATUS(STMT) do {} while (0);

std::string gstreamer_pipeline(int sensor_id, int capture_width,
    int capture_height, int display_width,
    int display_height, int framerate,
    int flip_method)
{
    return "nvarguscamerasrc sensor_id=" + std::to_string(sensor_id) + " ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" + std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) + "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, "
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      "format=(string)BGR ! appsink max-buffers=2 drop=true";
}


int main(int argc, char *argv[])
{
    int retval = 0;

    // OpenCV objects
    cv::Mat cvImageLeft, cvImageRight;
    cv::Mat cvDisparity;

    // VPI objects
    VPIStream stream       = NULL;
    VPIPayload stereo      = NULL;

    VPIImage imgL         = NULL;
    VPIImage imgR         = NULL;
    VPIImage imgL_8u      = NULL;
    VPIImage imgR_8u      = NULL;
    VPIImage imgL_8u_rect = NULL;
    VPIImage imgR_8u_rect = NULL;
    VPIImage imgL_270p    = NULL;
    VPIImage imgR_270p    = NULL;
    VPIImage disparity     = NULL;
    VPIImage confidenceMap = NULL;

    // Image params
    int32_t W_input  = 1920;
    int32_t H_input  = 1080;
    int32_t W_stereo  = 480;
    int32_t H_stereo = 270;
    int32_t FPS = 30;
    int32_t FLIP_METHOD = 2;

    // Create camera capture pipelines
    std::string pipeline_right = gstreamer_pipeline(1, W_input, H_input, W_input, H_input, FPS, FLIP_METHOD);
    std::string pipeline_left = gstreamer_pipeline(0, W_input, H_input, W_input, H_input, FPS, FLIP_METHOD);

    cv::VideoCapture cap_l(pipeline_left, cv::CAP_GSTREAMER);
    cv::VideoCapture cap_r(pipeline_right, cv::CAP_GSTREAMER);

    if (!cap_l.isOpened() || !cap_r.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return (-1);
    }

    // Create VPI stream
    CHECK_STATUS(vpiStreamCreate(0, &stream));

    // Format conversion parameters needed for input pre-processing
    VPIConvertImageFormatParams convParams;
    CHECK_STATUS(vpiInitConvertImageFormatParams(&convParams));

    // Set algorithm parameters to be used. Only values what differs from defaults will be overwritten.
    VPIStereoDisparityEstimatorCreationParams stereoParams;
    CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereoParams));
    stereoParams.maxDisparity = 256;
    VPIImageFormat stereoFormat = VPI_IMAGE_FORMAT_Y16_ER; // 16bpp format

    // Allocate input to stereo disparity algorithm, pitch-linear 16bpp grayscale
    CHECK_STATUS(vpiImageCreate(W_input, H_input, stereoFormat, 0, &imgL_8u));
    CHECK_STATUS(vpiImageCreate(W_input, H_input, stereoFormat, 0, &imgR_8u));
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, stereoFormat, 0, &imgL_270p));
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, stereoFormat, 0, &imgR_270p));
    CHECK_STATUS(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, W_stereo, H_stereo, stereoFormat, &stereoParams, &stereo)); // create stereo estimator object
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, VPI_IMAGE_FORMAT_U16, 0, &disparity)); // create disparity buffer
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap)); // create confidence buffer

    // Read cameras once to create a non-empty cv::Mat
    cap_l.read(cvImageLeft);
    cap_r.read(cvImageRight);
    // Wrap cv::Mat in vpiImage
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageLeft, 0, &imgL));
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageRight, 0, &imgR));

    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<1000; ++i)
    {        
        // Read images. They are 270p U8 arrays
        cap_l.read(cvImageLeft);
        cap_r.read(cvImageRight);
        std::cout << "Pre" << std::endl;
        vpiImageSetWrappedOpenCVMat(imgL, cvImageLeft);
        vpiImageSetWrappedOpenCVMat(imgL, cvImageRight);
        std::cout << "Post" << std::endl;

        // Convert to uint8
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgL, imgL_8u, &convParams));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgR, imgR_8u, &convParams));
        // Rescale
        CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_CUDA, imgL_8u, imgL_270p, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
        CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_CUDA, imgR_8u, imgR_270p, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
        // Disparity
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, stereo, imgL_270p, imgR_270p, disparity,
                                                        confidenceMap, NULL));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Lock output to retrieve its data on cpu memory
        VPIImageData data;
        CHECK_STATUS(vpiImageLock(disparity, VPI_LOCK_READ, &data));

        // Make an OpenCV matrix out of this image
        CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvDisparity));


        // Scale result and write it to disk. Disparities are in Q10.5 format,
        // so to map it to float, it gets divided by 32. Then the resulting disparity range,
        // from 0 to stereo.maxDisparity gets mapped to 0-255 for proper output.
        cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 / (32 * stereoParams.maxDisparity), 0);

        // Apply JET colormap to turn the disparities into color, reddish hues
        // represent objects closer to the camera, blueish are farther away.
        cv::Mat cvDisparityColor;
        applyColorMap(cvDisparity, cvDisparityColor, cv::COLORMAP_JET);

        // Done handling output, don't forget to unlock it.
        CHECK_STATUS(vpiImageUnlock(disparity));
        
        cv::imshow("disparity_map.png", cvDisparityColor);
        cv::waitKey(1);
        // std::cout << i << std::endl;

        auto end = std::chrono::steady_clock::now();

        auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Frame" << std::to_string(i) << " " << float(i)/elapsed_time_ms*1000 << "FPS" << std::endl;

    }




    // ========
    // Clean up

    // Destroying stream first makes sure that all work submitted to
    // it is finished.
    vpiStreamDestroy(stream);

    // Only then we can destroy the other objects, as we're sure they
    // aren't being used anymore.

    vpiImageDestroy(imgL);
    vpiImageDestroy(imgR);
    vpiImageDestroy(imgL_8u);
    vpiImageDestroy(imgR_8u);
    vpiImageDestroy(imgL_270p);
    vpiImageDestroy(imgR_270p);
    vpiImageDestroy(confidenceMap);
    vpiImageDestroy(disparity);
    vpiPayloadDestroy(stereo);

    return retval;
}

// vim: ts=8:sw=4:sts=4:et:ai
