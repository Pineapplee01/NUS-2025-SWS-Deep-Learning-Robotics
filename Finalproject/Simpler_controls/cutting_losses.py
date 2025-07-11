'''
We decided to cut our losses during rapid prototyping and go for simpler controls so that we at least have a proof of Visual Servoing
Made in like... 30 mins, tested for 12 hours
Treating the navigation as a 1-D problem
Listens to the camera csv stream and passes them onto the pi through sockets after stripping them into:

offset: The x-coordinate distance between the centre of camera and the boundary box's centre for the detected object
box_width: width of the detected object's boundary box in pixels
box_height: height of the detected object's boundary box in pixels (useful to calculate the stopping point to deploy the pincers)

Used UDP instead of TCP here because the last time, the camera latency was abysmal
'''
import socket
import time
import serial

# === Serial setup ===
UDP_IP = "0.0.0.0"       # 监听所有网卡
UDP_PORT = 8888

# === Camera setup ===
max_width = 640
max_height = 480
#The camera is set up to send data in the format "offset,width,height"

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT}...")

try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.0001)
    print("Serial connection established")
except Exception as e:
    print(f"Serial connection failed: {e}")
    ser = None    

while True:
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode().strip()
        offset_str, width_str, height_str = message.split(',')
        
        offset = int(offset_str) # box offset from center already calulated 
        box_width = int(width_str)
        box_height = int(height_str)

        print(f"offset: {offset}, box_width: {box_width}, box_height: {box_height}")

        ser.flush() 

        if offset > 20:
            cmd = 'D'
        elif offset < -20:
            cmd = 'A'
        elif box_width < max_width and box_height < 0.65 * max_height:
            cmd = 'W'
        else:
            cmd = 'X'
            ser.write((cmd + '\n').encode())
            print("Opening gripper")
            ser.write(('U' +'\n').encode())
            ser.write(('U' +'\n').encode())
            ser.write(('U' +'\n').encode())
            time.sleep(5)
            print("Closing gripper")
            ser.write(('N' +'\n').encode())
            ser.write(('N' +'\n').encode())
            ser.write(('N' +'\n').encode())

        ser.write((cmd + '\n').encode())
        print(f"Sent: {cmd}")
        time.sleep(0.1)                  

    except KeyboardInterrupt:
        print("Stopped")
        ser.close()

    except Exception as e:
        print(f"wrong:{e}")    
