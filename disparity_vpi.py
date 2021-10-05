import time
import numpy as np
import vpi
import cv2
from threading import Thread
from PIL import Image
from jetvision.elements import Camera


MAX_DISP = 64
WINDOW_SIZE = 10


def get_calibration() -> tuple:
    fs = cv2.FileStorage(
        "calibration/rectify_map_imx219_160deg_1080p.yaml", cv2.FILE_STORAGE_READ
    )

    map_l = (fs.getNode("map_l_x").mat(), fs.getNode("map_l_y").mat())

    map_r = (fs.getNode("map_r_x").mat(), fs.getNode("map_r_y").mat())

    fs.release()

    return map_l, map_r


def make_vpi_warpmap(cv_maps) -> vpi.WarpMap:
    
    src_map = cv_maps[0]
    idk_what_that_is = cv_maps[1]
    map_y, map_x = src_map[:,:,0], src_map[:,:,1]
    
    warp = vpi.WarpMap(vpi.WarpGrid((1920,1080)))
    # (H, W, C) -> (C,H,W)
    arr_warp = np.asarray(warp)
    wx = arr_warp[:,:,0]
    wy = arr_warp[:,:,1]

    wy[:1080,:] = map_x
    wx[:1080,:] = map_y
    
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


if __name__ == "__main__":

    map_l, map_r = get_calibration()
    warp_l = make_vpi_warpmap(map_l)
    warp_r = make_vpi_warpmap(map_r)

    cam_l = CameraThread(1)
    cam_r = CameraThread(0)

    try:
        for i in range(3000):

            ts = []
            ts.append(time.perf_counter())
            # confidenceMap = vpi.Image(vpi_l.size, vpi.Format.U16)

            # Read Images
            arr_l = cam_l.image
            arr_r = cam_r.image
            ts.append(time.perf_counter())

            # Convert to VPI image
            vpi_l = vpi.asimage(arr_l)
            vpi_r = vpi.asimage(arr_r)
            ts.append(time.perf_counter())

            # Rectify
            with vpi.Backend.CUDA:
                vpi_l = vpi_l.remap(warp_l)
                vpi_r = vpi_r.remap(warp_r)
            ts.append(time.perf_counter())

            # Resize
            with vpi.Backend.CUDA:
                vpi_l = vpi_l.rescale((480, 270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                vpi_r = vpi_r.rescale((480, 270), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            ts.append(time.perf_counter())

            # Convert to 16bpp
            with vpi.Backend.CUDA:
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

            with vpi.Backend.CUDA:
                # Convert again
                disparity_8bpp = disparity_16bpp.convert(
                    vpi.Format.U8, scale=255.0 / (32 * MAX_DISP)
                )
            ts.append(time.perf_counter())

            # CPU mapping
            disp_arr = disparity_8bpp.cpu()
            ts.append(time.perf_counter())

            # Colormap
            # disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
            ts.append(time.perf_counter())

            # Render
            # cv2.imshow("Disparity", disp_arr)
            # cv2.waitKey(1)
            ts.append(time.perf_counter())

            ts = np.array(ts)
            ts_deltas = np.diff(ts)


            debug_str = f"Iter {i}\n"

            for task, dt in zip(
                [
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
                ],
                ts_deltas,
            ):
                debug_str += f"{task} {1000*dt:0.2f}\n"

            debug_str += f"Sum: {sum(ts_deltas)}\n\n"

            del vpi_l, vpi_r, arr_l, arr_r
            del disparity_8bpp, disparity_16bpp, vpi_l_16bpp, vpi_r_16bpp

            print(debug_str)

            if i % 30 == 0:
                vpi.clear_cache()

    except KeyboardInterrupt as e:
        print(e)
    finally:
        cam_l.stop()
        cam_r.stop()
