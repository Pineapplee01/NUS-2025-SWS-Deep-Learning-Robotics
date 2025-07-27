'''
We decided to cut our losses during rapid prototyping and go for simpler controls so that we at least have a proof of Visual Servoing
Made in like... 30 mins, tested and iterated over for 57 hours
Treating the navigation as a 1-D problem
Listens to the camera csv stream and passes them onto the pi through sockets after stripping them into:

offset: The x-coordinate distance between the centre of camera and the boundary box's centre for the detected object
box_width: width of the detected object's boundary box in pixels
box_height: height of the detected object's boundary box in pixels (useful to calculate the stopping point to deploy the pincers)

Used UDP instead of TCP here because the last time, the camera latency was abysmal
This gives the best performance till now, will add more later.

Now the robot operates in phases (state machines, yay! ToC is helpful)
It first searches for a target object, then turns around till it detects its next destination (here 'milo', expected 'human') to receive
the delivery
'''

import socket
import serial
import time

# === CONFIG ===
SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 115200
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
SERIAL_TIMEOUT = 0.05

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.05)

ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
time.sleep(2)
print("[CONTROL] Serial connected.")

# === STATE FLAGS ===
clamped = False
delivered = False
last_cmd = 'X'

searching = True     # looking for object
approaching = False  # approaching object
delivering = False   # looking for human after grabbing

trig_object = False
trig_human = False

# === MAIN LOOP ===
while True:
    # === Check Arduino ===
    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[SERIAL] {line}")
            if "Clamped" in line:
                clamped = True
                last_cmd = 'X'
                ser.write((last_cmd + '\n').encode())
                print("[CONTROL] Clamped. Now search for human.")
                delivering = True
                searching = False
                approaching = False

            elif "Delivered" in line:
                delivered = True
                last_cmd = 'X'
                ser.write((last_cmd + '\n').encode())
                print("[CONTROL] Delivered to human. All done.")
                break

    # === Check Camera ===
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode().strip()
        print(f"[CONTROL] Camera: {message}")

        offset_str, width_str, height_str, object_str = message.split(',')
        offset = int(offset_str)
        box_width = int(width_str)
        box_height = int(height_str)
        object_type = object_str.strip()

        object_detected = (box_width > 40 or box_height > 40)
        trig_object = (object_type == "cup") and object_detected
        trig_human = (object_type == "milo") and object_detected

        if clamped and not delivering:
            last_cmd = 'X'

        elif searching:
            if trig_object:
                print("[CONTROL] Object detected! Switching to approach mode.")
                searching = False
                approaching = True

                if offset > 25:
                    last_cmd = 'D'
                elif offset < -25:
                    last_cmd = 'A'
                else:
                    last_cmd = 'W'
            else:
                last_cmd = 'A'  # rotate left if nothing found

        elif approaching:
            if trig_object:
                if offset > 25:
                    last_cmd = 'D'
                elif offset < -25:
                    last_cmd = 'A'
                else:
                    last_cmd = 'W'
            else:
                print("[CONTROL] Lost object, resuming search.")
                approaching = False
                searching = True
                last_cmd = 'A'

        elif delivering:
            if trig_human:
                print("[CONTROL] Milo detected! Approaching to deliver.")
                if offset > 25:
                    last_cmd = 'D'
                elif offset < -25:
                    last_cmd = 'A'
                else:
                    last_cmd = 'W'
                    if box_width > 40 and box_height > 115:
                        print("[CONTROL] Delivered to milo.") #placeholder for human
                        delivered = True
                        last_cmd = 'N' #unclamp
                        ser.write((last_cmd + '\n').encode())
                        last_cmd = 'X'  # stop after delivery
                        ser.write((last_cmd + '\n').encode())
                        delivering = False
            else:
                last_cmd = 'A'  # rotate to find 'human'

        ser.write((last_cmd + '\n').encode())
        print(f"[SEND] Sent: {last_cmd}")

    except socket.timeout:
        pass

    time.sleep(0.001)
