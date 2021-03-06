import processe
import cv2
import json
import numpy as np
from networktables import NetworkTables
import constants
import logging
import sys


def main():
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) != 2:
         print("err")
         exit(0)
    ip = sys.argv[1]
    # As a client to connect to a robot
    print(NetworkTables.initialize(server=ip))
    while not NetworkTables.isConnected():
         print(NetworkTables.initialize(server=ip))
    sd = NetworkTables.getTable('Vision')

    # load vision data
    data = processe.newest_save()
    light = data["light"]
    blur = data["blur"]
    min_hsv = np.array(data["min"])
    max_hsv = np.array(data["max"])
    rotation = processe.get_rotation_matrix(np.array(data["rotation"]))

    # camera configuration
    cap = cv2.VideoCapture(0)
    cap.set(15, light)

    while True:
        _, frame = cap.read()
        frame = processe.get_image(frame, rotation)
        angle, distance, processed = processe.get_vision_data(frame, min_hsv, max_hsv, blur, constants.STICKER_AREA)
        cv2.imshow("before", frame)
        cv2.imshow("after", processed)
        # print(angle)
        if distance is not None:
            velocity = processe.velocity(distance, 0.6)
            if velocity is not None:
                sd.putNumber('vel', float(velocity))
        if angle is not None:
            sd.putNumber('ang', float(angle))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
