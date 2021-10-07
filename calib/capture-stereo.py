import time
import cv2

from camera import Camera

if __name__ == "__main__":
    
    cam_l= Camera(0, flip=True)
    cam_r = Camera(1, flip=True)
    cnt = 0

    while True:

        img_l = cam_l.read()
        img_r = cam_r.read()
        
        print(img_l)
        
        cv2.imshow("windowname", img_l)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            cnt +=1
            cv2.imwrite(f"calib_pairs/right/{cnt:03d}.png", img_r)
            cv2.imwrite(f"calib_pairs/left/{cnt:03d}.png", img_l)
            print("pic!")
