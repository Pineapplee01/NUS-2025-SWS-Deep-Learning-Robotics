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

clamped = False
last_cmd = 'X'

searching = True     # scanning for object
approaching = False  # approaching object when found

while searching or approaching:
    # === Check Arduino ===
    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[SERIAL] {line}")
            if "Clamped" in line:
                clamped = True
                last_cmd = 'X'
                ser.write((last_cmd + '\n').encode())
                searching = False
                approaching = False
                print("[CONTROL] Clamped, stopping everything.")

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

        # Decide if object is detected
        object_detected = (box_width > 40 or box_height > 40)

        if clamped:
            last_cmd = 'X'

        elif searching:
            if object_detected:
                print("[CONTROL] Object detected! Switching to approach mode.")
                searching = False
                approaching = True

                # Use normal correction immediately
                if offset > 25:
                    last_cmd = 'D'
                elif offset < -25:
                    last_cmd = 'A'
                else:
                    last_cmd = 'W'

            else:
                # Object NOT detected → keep turning left slowly : doesn't work :(
                last_cmd = 'A'

        elif approaching:
            # Object in sight → correct and move forward
            if offset > 25:
                last_cmd = 'D'
            elif offset < -25:
                last_cmd = 'A'
            else:
                last_cmd = 'W'

        ser.write((last_cmd + '\n').encode())
        print(f"[SEND] Sent: {last_cmd}")

    except socket.timeout:
        pass

    time.sleep(0.001) #prevent busy waiting
