wget https://www.arducam.com/downloads/Jetson/Camera_overrides.tar.gz -O /tmp/camera_overrides.tar.gz
tar zxvf /tmp/camera_overrides.tar.gz --directory=/tmp/
sudo cp /tmp/camera_overrides.isp /var/nvidia/nvcam/settings/
sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
