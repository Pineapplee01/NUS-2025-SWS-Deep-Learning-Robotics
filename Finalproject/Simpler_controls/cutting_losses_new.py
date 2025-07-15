import socket
import serial
import time

# === Local UDP === needed to take this approach to enable auto multi-threading by the pu

UDP_IP = "0.0.0.0"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for local UDP on {UDP_IP}:{UDP_PORT}")

# === Serial ===
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
print("Serial connected.")

cmd ='X' #default command to stop the robot

trig_object = True  # Flag to indicate if object detction in progress
trig_human = False  # Flag to indicate if human detection in progress

# === Main loop ===

while trig_object:
    # === 1) Check Arduino ===
    if ser.in_waiting:
        line = ser.readline().decode().strip()
        print(f"Arduino says: {line}")
        if "Clamped" in line:
            print("Clamp done")
            cmd = 'X'
            ser.write((cmd + '\n').encode())
            print(f"Sent: {cmd}")
            time.sleep(3)
            trig_object = False
            trig_human = True

    sock.settimeout(0.1)
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode().strip()
        print(f"Got local UDP: {message}")

        offset_str, width_str, height_str = message.split(',')
        offset = int(offset_str)
        width = int(width_str)
        height = int(height_str)

        # Decide command if not already clamped (KINDA)
        if offset > 20:
            cmd = 'D'
        elif offset < -20:
            cmd = 'A'
        elif cmd!= 'X':
            cmd = 'W'
        else:
            cmd = 'A'    

        ser.write((cmd + '\n').encode())
        print(f"Sent: {cmd}")

    except socket.timeout:
        pass

    time.sleep(0.05)
#will add fucntionality of what happens when a human is detected later
