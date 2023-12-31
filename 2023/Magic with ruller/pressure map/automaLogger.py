import serial
import os

LOG_TO_CONSOLE = False
BAUD = 115200
PORT = 'COM4'
ROOT = ".\OpenCV and tracers/2023/Magic with ruller/pressure map/logs/"

filename = input("enter file name: ")
if os.path.exists(ROOT+filename+".txt"):
    if input("this file alredy exists and will bi overwriten, do you want to continue& (y/n): ") != "y":
        print("cancelled")
        import sys
        sys.exit(0)
log = open(ROOT+filename+".txt", "wb")

sp = serial.Serial(PORT, BAUD, timeout=1)

try:
    print("logging started")
    while True:
        line = sp.readline()
        if LOG_TO_CONSOLE:
            print(line)
        log.write(line)
except KeyboardInterrupt:
    print("interrupted")
finally:
    print("logging finished")
    log.close()
    sp.close()
