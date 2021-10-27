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

int main(int argc, char *argv[])
{
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvImageLeft, cvImageRight;
    cv::Mat cvDisparity;

    uint32_t backends = VPI_BACKEND_CUDA;

    // VPI objects that will be used
    VPIImage inLeft        = NULL;
    VPIImage inRight       = NULL;
    VPIImage tmpLeft       = NULL;
    VPIImage tmpRight      = NULL;
    VPIImage stereoLeft    = NULL;
    VPIImage stereoRight   = NULL;
    VPIImage disparity     = NULL;
    VPIImage confidenceMap = NULL;
    VPIStream stream       = NULL;
    VPIPayload stereo      = NULL;

    int retval = 0;

    // Load the input images
    cvImageLeft = cv::imread("chair_stereo_left.png");
    cvImageRight = cv::imread("chair_stereo_right.png");

    int32_t W_in  = cvImageLeft.cols;
    int32_t H_in = cvImageLeft.rows;
    int32_t W_out  = W_in;
    int32_t H_out = H_in;
    int32_t MAX_DISPARITY = 256;

    // Create VPI stream
    CHECK_STATUS(vpiStreamCreate(0, &stream));

    // We now wrap the loaded images into a VPIImage object to be used by VPI.
    // VPI won't make a copy of it, so the original image must be in scope at all times.
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageLeft, 0, &inLeft));
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageRight, 0, &inRight));

    // Format conversion parameters needed for input pre-processing
    VPIConvertImageFormatParams convParams;
    CHECK_STATUS(vpiInitConvertImageFormatParams(&convParams));

    // Set algorithm parameters to be used. Only values what differs from defaults will be overwritten.
    VPIStereoDisparityEstimatorCreationParams stereoParams;
    CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&stereoParams));

    // Define some backend-dependent parameters

    stereoParams.maxDisparity = MAX_DISPARITY;

    VPIImageFormat stereoFormat;
    stereoFormat = VPI_IMAGE_FORMAT_Y16_ER; // 16bpp format

    CHECK_STATUS(vpiCreateStereoDisparityEstimator(backends, W_out, H_out, stereoFormat, &stereoParams, &stereo)); // create stereo estimator object
    CHECK_STATUS(vpiImageCreate(W_out, H_out, VPI_IMAGE_FORMAT_U16, 0, &disparity)); // create disparity buffer
    CHECK_STATUS(vpiImageCreate(W_out, H_out, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap)); // create confidence buffer


    // Allocate input to stereo disparity algorithm, pitch-linear 16bpp grayscale
    CHECK_STATUS(vpiImageCreate(W_out, H_out, stereoFormat, 0, &stereoLeft));
    CHECK_STATUS(vpiImageCreate(W_out, H_out, stereoFormat, 0, &stereoRight));

    // ================
    // Processing stage

    // -----------------
    // Pre-process input

    // Convert opencv input to grayscale format using CUDA

    
    auto start = std::chrono::steady_clock::now();

    for(int i=0; i<1000; ++i)
    {

        // CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_CUDA, input, output, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
        // CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_CUDA, input, output, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));

        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inLeft, stereoLeft, &convParams));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, inRight, stereoRight, &convParams));
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, stereo, stereoLeft, stereoRight, disparity,
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
        
        // cv::imshow("disparity_map.png", cvDisparityColor);
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

    vpiImageDestroy(inLeft);
    vpiImageDestroy(inRight);
    vpiImageDestroy(tmpLeft);
    vpiImageDestroy(tmpRight);
    vpiImageDestroy(stereoLeft);
    vpiImageDestroy(stereoRight);
    vpiImageDestroy(confidenceMap);
    vpiImageDestroy(disparity);
    vpiPayloadDestroy(stereo);

    return retval;
}

// vim: ts=8:sw=4:sts=4:et:ai
