import time
import queue
import numpy as np
import cv2
from threading import Thread
from jetvision.elements import Camera
from concurrent.futures import ThreadPoolExecutor

MAX_DISP = 128
WINDOW_SIZE = 10


class CAM_PARAMS:
    SENSOR_FULL_X = 3.68  # mm
    SENSOR_FULL_Y = 2.76  # mm
    SENSOR_FULL_X_PX = 3280  # px
    SENSOR_FULL_Y_PX = 2464  # px

    PIXEL_SIZE = SENSOR_FULL_X / 3280  # approx. 0.00112mm (1.12 um) for IMX219
    F_PX = 1570  # focal length in pixels (from intrinsic calibration)
    F = F_PX * PIXEL_SIZE
    B = 100  # mm (baseline)


class Depth(Thread):
    def __init__(self):

        super().__init__()
        print("Reading camera calibration...")
        self._map_l, self._map_r = get_calibration()
        self._cam_r = CameraThread(0)
        self._cam_l = CameraThread(1)
        self._disp_arr = None
        self._should_run = True
        self._dq = queue.deque(maxlen=3)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.start()

        # Wait for the deque to start filling up
        while len(self._dq) < 1:
            time.sleep(0.1)

    def stop(self):
        self._should_run = False
        self._cam_l.stop()
        self._cam_r.stop()

    def disparity(self):
        while len(self._dq) == 0:
            time.sleep(0.01)
        return self._dq.pop()

    def enqueue_async(self, vpi_image):
        arr = vpi_image.cpu().copy()
        self._dq.append(arr)
        del vpi_image

    def run(self):
        import vpi  # Don't import vpi in main thread to avoid creating a global context

        i = 0
        self._warp_l = make_vpi_warpmap(self._map_l)
        self._warp_r = make_vpi_warpmap(self._map_r)
        ts_history = []

        while self._should_run:
            i += 1
            ts = []
            ts.append(time.perf_counter())

            with vpi.Backend.CUDA:
                ts.append(time.perf_counter())
                # confidenceMap = vpi.Image(vpi_l.size, vpi.Format.U16)

                # Read Images
                arr_l = self._cam_l.image
                arr_r = self._cam_r.image
                ts.append(time.perf_counter())

                # Convert to VPI image
                vpi_l = vpi.asimage(arr_l)
                vpi_r = vpi.asimage(arr_r)
                ts.append(time.perf_counter())

                # Rectify
                vpi_l = vpi_l.remap(self._warp_l)
                vpi_r = vpi_r.remap(self._warp_r)
                ts.append(time.perf_counter())

                # Resize
                vpi_l = vpi_l.rescale(
                    (480, 270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO
                )
                vpi_r = vpi_r.rescale(
                    (480, 270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO
                )
                ts.append(time.perf_counter())

                # Convert to 16bpp
                vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
                vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
                ts.append(time.perf_counter())

                # Disparity
                disparity_16bpp = vpi.stereodisp(
                    vpi_l_16bpp,
                    vpi_r_16bpp,
                    out_confmap=None,
                    backend=vpi.Backend.CUDA,
                    window=WINDOW_SIZE,
                    maxdisp=MAX_DISP,
                )
                ts.append(time.perf_counter())

                # Convert again
                disparity_8bpp = disparity_16bpp.convert(
                    vpi.Format.U8, scale=255.0 / (32 * MAX_DISP)
                )
                ts.append(time.perf_counter())

                # CPU mapping
                # self._disp_arr = disparity_8bpp
                self._executor.submit(self.enqueue_async, disparity_8bpp)
                # global frames
                # frames.append(self._disp_arr)
                ts.append(time.perf_counter())

                # Render
                # cv2.imshow("Disparity", disp_arr)
                # cv2.waitKey(1)
                ts.append(time.perf_counter())

                ts = np.array(ts)
                ts_deltas = np.diff(ts)

                ts_history.append(ts_deltas)

                debug_str = f"Iter {i}\n"

            tasks = [
                "Enter context",
                "Read images",
                "Convert to VPI image",
                "Rectify",
                "Resize 1080p->270p",
                "Convert to 16bpp",
                "Disparity",
                "Convert 16bpp->8bpp",
                ".cpu() mapping",
                "OpenCV colormap",
                "Render",
            ]

            debug_str += f"Sum: {sum(ts_deltas):0.2f}\n\n"
            task_acc_time = 1000 * np.array(ts_history).mean(axis=0)

            for task, dt in zip(tasks, task_acc_time):
                debug_str += f"{task.ljust(20)} {dt:0.2f}\n"

            # print(debug_str)

            if i % 10 == 0:
                vpi.clear_cache()


def disp2depth(disp_arr):
    F, B, PIXEL_SIZE = (CAM_PARAMS.F, CAM_PARAMS.B, CAM_PARAMS.PIXEL_SIZE)
    disp_arr[disp_arr < 10] = 10
    disp_arr_mm = disp_arr * PIXEL_SIZE  # convert disparites from px -> mm
    depth_arr_mm = F * B / (disp_arr_mm + 1e-10)  # calculate depth
    return depth_arr_mm


def get_calibration() -> tuple:
    fs = cv2.FileStorage(
        "calibration/rectify_map_imx219_160deg_1080p.yaml", cv2.FILE_STORAGE_READ
    )
    map_l = (fs.getNode("map_l_x").mat(), fs.getNode("map_l_y").mat())
    map_r = (fs.getNode("map_r_x").mat(), fs.getNode("map_r_y").mat())
    fs.release()

    return map_l, map_r


def make_vpi_warpmap(cv_maps):
    import vpi

    src_map = cv_maps[0]
    idk_what_that_is = cv_maps[1]
    map_y, map_x = src_map[:, :, 0], src_map[:, :, 1]

    warp = vpi.WarpMap(vpi.WarpGrid((1920, 1080)))
    # (H, W, C) -> (C,H,W)
    arr_warp = np.asarray(warp)
    wx = arr_warp[:, :, 0]
    wy = arr_warp[:, :, 1]

    wy[:1080, :] = map_x
    wx[:1080, :] = map_y

    return warp


class CameraThread(Thread):
    def __init__(self, sensor_id) -> None:

        super().__init__()
        self._camera = Camera(sensor_id)
        self._should_run = True
        self._image = self._camera.read()
        self.start()

    def run(self):
        while self._should_run:
            self._image = self._camera.read()

    @property
    def image(self):
        # NOTE: if we care about atomicity of reads, we can add a lock here
        return self._image

    def stop(self):
        self._should_run = False
        self._camera.stop()


def calculate_depth(disp_arr):
    F, B, PIXEL_SIZE = (CAM_PARAMS.F, CAM_PARAMS.B, CAM_PARAMS.PIXEL_SIZE)
    disp_arr_mm = disp_arr * PIXEL_SIZE  # disparites in mm
    depth_arr_mm = F * B / (disp_arr_mm + 1e-1)  # distances
    return depth_arr_mm


if __name__ == "__main__":

    DISPLAY = True
    SAVE = True
    frames_d = []
    frames_rgb = []

    depth = Depth()
    t1 = time.perf_counter()

    for i in range(100):
        disp_arr = depth.disparity()
        frames_d.append(disp_arr)
        frames_rgb.append(depth._cam_l.image)
        print(i)
        # print(disp_arr)
        # depth_arr = calculate_depth(disp_arr)
        # print(f"{i}/20: {disp_arr:0.2f}")
        # time.sleep(1)

        if DISPLAY:
            disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
            cv2.imshow("Depth", disp_arr)
            cv2.imshow("Image", cv2.resize(depth._cam_l.image, (480, 270)))
            cv2.waitKey(1)

    depth.stop()
    t2 = time.perf_counter()
    print(f"Approx framerate: {len(frames_d)/(t2-t1)} FPS")

    # Save frames
    for i, (disp_arr, rgb_arr) in enumerate(zip(frames_d, frames_rgb)):
        print(f"{i}/{len(frames_d)}", end="\r")
        disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
        cv2.imwrite(f"images/depth_{i}.jpg", disp_arr)
        cv2.imwrite(f"images/rgb_{i}.jpg", rgb_arr)
