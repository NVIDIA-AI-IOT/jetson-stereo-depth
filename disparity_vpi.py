import time
import numpy as np
import vpi
import cv2
from threading import Thread
from PIL import Image
from jetvision.elements import Camera


MAX_DISP=64


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


cam_l = CameraThread(0)
cam_r = CameraThread(1)

try:

    with vpi.Backend.CUDA:
        for i in range(100):
            # confidenceMap = vpi.Image(vpi_l.size, vpi.Format.U16)
            #
            arr_l = cam_l.image
            arr_r = cam_r.image

            arr_l = cv2.cvtColor(arr_l, cv2.COLOR_RGB2GRAY)
            arr_r = cv2.cvtColor(arr_r, cv2.COLOR_RGB2GRAY)

            arr_l = cv2.resize(arr_l, (480, 270))
            arr_r = cv2.resize(arr_r, (480, 270))

            vpi_l = vpi.asimage(arr_l)
            vpi_r = vpi.asimage(arr_r)

            vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
            vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)


            st = time.perf_counter()
            vpi_l_16bpp = vpi_l.convert(vpi.Format.U16, scale=1)
            vpi_r_16bpp = vpi_r.convert(vpi.Format.U16, scale=1)    
            
            disparity_16bpp = vpi.stereodisp(vpi_l_16bpp, vpi_r_16bpp, out_confmap=None, backend=vpi.Backend.CUDA, window=5, maxdisp=MAX_DISP)
            disparity_8bpp = disparity_16bpp.convert(vpi.Format.U8, scale=255.0/(32*MAX_DISP))

            disp_arr = disparity_8bpp.cpu()
            cv2.imshow("Disparity", disp_arr)

            print(i)

            # print(time.perf_counter() - st)
except KeyboardInterrupt as e:
    print(e)
finally:
    cam_l.stop()
    cam_r.stop()
