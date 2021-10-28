#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp> // imshow()
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

void errIfNotSuccess(VPIStatus status)
{
    if (status != VPI_SUCCESS)
    {
        char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
        vpiGetLastStatusMessage(buffer, sizeof(buffer));
        std::ostringstream ss;
        ss << vpiStatusGetName(status) << ": " << buffer;
        throw std::runtime_error(ss.str());
    }
}

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

std::string gstreamer_pipeline(
    int sensor_id,
    int capture_width,
    int capture_height,
    int display_width,
    int display_height,
    int framerate,
    int flip_method
    )
{
    return "nvarguscamerasrc sensor_id=" + std::to_string(sensor_id) + " ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" + std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) + "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=2 drop=true";
}

int main(int argc, char *argv[])
{
    int retval = 0;

    // OpenCV objects
    cv::Mat cvImageLeft, cvImageRight;
    cv::Mat cvDisparity, cvDisparityColor;

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
    int32_t W_input  = 1280;
    int32_t H_input  = 720;
    int32_t W_stereo  = 480;
    int32_t H_stereo = 270;
    int32_t FPS = 60;
    int32_t FLIP_METHOD = 2;

    // Create camera capture pipelines
    std::string pipeline_right = gstreamer_pipeline(1, W_input, H_input, W_input, H_input, FPS, FLIP_METHOD);
    std::string pipeline_left = gstreamer_pipeline(0, W_input, H_input, W_input, H_input, FPS, FLIP_METHOD);
    cv::VideoCapture cap_l(pipeline_left, cv::CAP_GSTREAMER);
    cv::VideoCapture cap_r(pipeline_right, cv::CAP_GSTREAMER);

    // Ensure both cameras are opened
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

    // Allocate buffers in Y16 format
    VPIImageFormat inputFormat = VPI_IMAGE_FORMAT_Y16_ER;
    VPIStatus status;
    status = vpiImageCreate(W_input, H_input, inputFormat, 0, &imgL_8u);
    errIfNotSuccess(status);
    CHECK_STATUS(vpiImageCreate(W_input, H_input, inputFormat, 0, &imgR_8u));
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, inputFormat, 0, &imgL_270p));
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, inputFormat, 0, &imgR_270p));

    // create stereo estimator object + disparity buffer + confidence buffer
    CHECK_STATUS(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, W_stereo, H_stereo, inputFormat, &stereoParams, &stereo));
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, VPI_IMAGE_FORMAT_U16, 0, &disparity));
    CHECK_STATUS(vpiImageCreate(W_stereo, H_stereo, VPI_IMAGE_FORMAT_U16, 0, &confidenceMap));

    // Read cameras once to create a non-empty cv::Mat
    cap_l.read(cvImageLeft);
    cap_r.read(cvImageRight);

    // Wrap cv::Mat in vpiImage
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageLeft, 0, &imgL));
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageRight, 0, &imgR));

    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<300; ++i)
    {
        // Read images
        cap_l.read(cvImageLeft);
        cap_r.read(cvImageRight);
        vpiImageSetWrappedOpenCVMat(imgL, cvImageLeft);
        vpiImageSetWrappedOpenCVMat(imgL, cvImageRight);

        // Convert to uint8
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgL, imgL_8u, &convParams));
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgR, imgR_8u, &convParams));

        // Rescale
        CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_CUDA, imgL_8u, imgL_270p, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
        CHECK_STATUS(vpiSubmitRescale(stream, VPI_BACKEND_CUDA, imgR_8u, imgR_270p, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));

        // Disparity
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, stereo, imgL_270p, imgR_270p, disparity, confidenceMap, NULL));

        // Sync
        CHECK_STATUS(vpiStreamSync(stream));

        // Lock disparity
        VPIImageData data;
        CHECK_STATUS(vpiImageLock(disparity, VPI_LOCK_READ, &data));

        // vpi image -> cv::Mat
        CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvDisparity));

        // Q10.5 -> float
        cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 / (32 * stereoParams.maxDisparity), 0);
        applyColorMap(cvDisparity, cvDisparityColor, cv::COLORMAP_JET);

        // Unlock disparity
        CHECK_STATUS(vpiImageUnlock(disparity));

        // Visualize
        cv::imshow("Diparity", cvDisparityColor);
        cv::waitKey(1);

        // Print debug info
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "\rFrame " << std::to_string(i) << " " << float(i)/elapsed_time_ms*1000 << "FPS" << std::flush;

    }

    // Destroy stream
    vpiStreamDestroy(stream);

    // Destroy objects
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
