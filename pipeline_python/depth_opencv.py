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

    cam_l = CameraThread(1)
    cam_r = CameraThread(0)

    try:
        with vpi.Backend.CUDA:
            for i in range(100):

                ts = []
                ts.append(time.perf_counter())
                # confidenceMap = vpi.Image(vpi_l.size, vpi.Format.U16)

                arr_l = cam_l.image
                arr_r = cam_r.image
                ts.append(time.perf_counter())

                # RGB -> GRAY
                # arr_l = cv2.cvtColor(arr_l, cv2.COLOR_RGB2GRAY)
                # arr_r = cv2.cvtColor(arr_r, cv2.COLOR_RGB2GRAY)
                ts.append(time.perf_counter())

                # Rectify
                arr_l_rect = cv2.remap(arr_l, *map_l, cv2.INTER_LANCZOS4)
                arr_r_rect = cv2.remap(arr_r, *map_r, cv2.INTER_LANCZOS4)
                ts.append(time.perf_counter())

                # Resize
                arr_l_rect = cv2.resize(arr_l_rect, (480, 270))
                arr_r_rect = cv2.resize(arr_r_rect, (480, 270))
                ts.append(time.perf_counter())

                # Convert to VPI image
                vpi_l = vpi.asimage(arr_l_rect)
                vpi_r = vpi.asimage(arr_r_rect)

                vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
                vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)

                vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
                vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)
                ts.append(time.perf_counter())

                disparity_16bpp = vpi.stereodisp(
                    vpi_l_16bpp,
                    vpi_r_16bpp,
                    out_confmap=None,
                    backend=vpi.Backend.CUDA,
                    window=WINDOW_SIZE,
                    maxdisp=MAX_DISP,
                )
                disparity_8bpp = disparity_16bpp.convert(
                    vpi.Format.U8, scale=255.0 / (32 * MAX_DISP)
                )
                ts.append(time.perf_counter())

                disp_arr = disparity_8bpp.cpu()
                ts.append(time.perf_counter())

                disp_arr = cv2.applyColorMap(disp_arr, cv2.COLORMAP_TURBO)
                ts.append(time.perf_counter())

                cv2.imshow("Disparity", disp_arr)
                cv2.waitKey(1)
                ts.append(time.perf_counter())

                ts = np.array(ts)
                ts_deltas = np.diff(ts)

                debug_str = f"Iter {i}\n"

                for task, dt in zip(
                    [
                        "Read images",
                        "OpenCV RGB->GRAY",
                        "OpenCV Rectify",
                        "OpenCV 1080p->270p Resize",
                        "VPI conversions",
                        "Disparity calc",
                        ".cpu() mapping",
                        "OpenCV colormap",
                        "Render",
                    ],
                    ts_deltas,
                ):
                    debug_str += f"{task} {1000*dt:0.2f}\n"

                print(debug_str)

    except KeyboardInterrupt as e:
        print(e)
    finally:
        cam_l.stop()
        cam_r.stop()
