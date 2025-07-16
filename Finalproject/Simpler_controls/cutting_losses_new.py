import socket
import serial
import time

# === CONFIG ===
SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 115200
# Local
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

trig_object = True  # Flag to indicate if object detction in progress
trig_human = False  # Flag to indicate if human detection in progress

while trig_object:
    while ser.in_waiting:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[SERIAL] {line}")
            if "Clamped" in line:
                last_cmd = 'X'
                ser.write((last_cmd + '\n').encode())
                clamped = True

    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode().strip()
        print(f"[CONTROL] Camera: {message}")

        offset_str, width_str, height_str = message.split(',')
        offset = int(offset_str)

        if clamped:
            last_cmd = 'X'
            trig_object = False
            trig_human = True
        elif offset > 30:
            last_cmd = 'D'
        elif offset < -30:
            last_cmd = 'A'
        else:
            last_cmd = 'W'

        ser.write((last_cmd + '\n').encode())
        print(f"[SEND] Sent: {last_cmd}")

    except socket.timeout:
        pass

    time.sleep(0.001) #prevent busy waiting
