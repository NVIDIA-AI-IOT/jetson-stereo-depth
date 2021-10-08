import time
import cv2
import sys
import numpy as np

sys.path.append("..")
from camera import Camera

if __name__ == "__main__":
    
    cam_l= Camera(0)
    cam_r = Camera(1)
    cnt = 0

    try:
        while True:

            print("Capturing. Press spacebar to capture an image pair.%%")

            img_l = cam_l.read()
            img_r = cam_r.read()

            img_lr = np.hstack([img_l, img_r])
            img_lr = cv2.resize(img_lr, (1920, 1080//2))
            
            cv2.imshow("windowname", img_lr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                cnt +=1
                cv2.imwrite(f"/home/nx/jetson-stereo-depth/calib/calib_images2/right/{cnt:03d}.png", img_r)
                cv2.imwrite(f"/home/nx/jetson-stereo-depth/calib/calib_images2/left/{cnt:03d}.png", img_l)
                print("pic!")
    except KeyboardInterrupt as e:
        print("closing")
    finally:
        cam_l.stop()
        cam_r.stop()