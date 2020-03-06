cd /home/pi/InfiniteRecharge2020-Vision/
sudo v4l2-ctl --set-ctrl=exposure_auto=1
sudo v4l2-ctl --set-ctrl=exposure_absolute=$(jq .light ./InfiniteRecharge2020-Vision/default.json)
sudo python3 NetworkTables.py