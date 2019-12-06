import time
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging
logging.basicConfig(level=logging.DEBUG)


NetworkTables.setIPAddress("10.XX.XX.XX")  # Change the address to your own
NetworkTables.setClientMode()
NetworkTables.initialize()

sd = NetworkTables.getTable("SmartDashboard")

i = 0
while True:
    try:
        print('robotTime:', sd.getNumber('robotTime'))
    except KeyError:
        print('robotTime: N/A')

    sd.putNumber('dsTime', i)
    time.sleep(1)
    i += 1