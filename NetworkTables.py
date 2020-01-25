import processe
import cv2
import json
import numpy as np
from networktables import NetworkTables


def main():
    # As a client to connect to a robot
    NetworkTables.initialize(server='10.82.23.2')
    sd = NetworkTables.getTable('SmartDashboard')

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

    while 69:
        _, frame = cap.read()
        frame = processe.get_image(frame, rotation)
        angle, processed = processe.get_center(frame, min_hsv, max_hsv, blur)
        cv2.imshow("before", frame)
        cv2.imshow("after", processed)
        if a is not None:
            sd.putNumber('ang', float(angle))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()