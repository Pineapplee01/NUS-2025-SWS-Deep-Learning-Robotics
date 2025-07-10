# pi_udp_receiver.py
import socket
import time
import serial

# === Serial setup ===
UDP_IP = "0.0.0.0"       # 监听所有网卡
UDP_PORT = 8888

# === Camera setup ===
max_width = 640
max_height = 480
#Ensure the camera is set up to send data in the format "offset,width,height"

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT}...")

# === PID class ===
class PiD:
    #constructor
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.prev_error = 0 # history tracker
        self.integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt # Integral term, incremental error accumulation
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative #formula applied discretely
        self.prev_error = error #update previous error for next iteration

        return output

# === Serial setup ===
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.0001) 

# === PID setup ===
heading_pid = PiD(Kp=0.01, Ki=0.0, Kd=0.002, setpoint=0)  # Tuning required
speed_pid = PiD(Kp=1.5, Ki=0.1, Kd=0.2, setpoint= 0.85*max_height)  # Tuning required

# === Robot parameters ===
base_speed = 230 #normal drive  
max_turn = 120   

last_time = time.time()

while True:
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode().strip()
        offset_str, width_str, height_str = message.split(',')
        offset = int(offset_str) # box offset from center already calulated 
        box_width = int(width_str)
        box_height = int(height_str)

        print(f"addr:{addr}")
        print (f"raw_msg:{message}")
        print(f"offset: {offset}, box_width: {box_width}, box_height: {box_height}")


        # === Get error ===
        # Get camera input here: ask Hammer
        heading_error = offset #<-- replaced with whatever calculated above (box offset)
        speed_error =  speed_pid.setpoint - box_height #<-- replace with whatever calculated from ablove (box size)

        # === Time loop update ===
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        #correction
        turn_correction = heading_pid.update(heading_error, dt)
        speed_correction = speed_pid.update(speed_error, dt)

        #limits
        turn_correction = max(-max_turn, min(max_turn, turn_correction)) #decide these values
        speed_correction = max(-base_speed, min(base_speed, speed_correction))

        # === Compute motor speeds ===
        left_speed = speed_correction - turn_correction
        right_speed = speed_correction + turn_correction

        # Ensure speeds are within bounds
        left_speed = max(0, min(255, left_speed))
        right_speed = max(0, min(255, right_speed))

        print(f"Left: {left_speed:.1f} Right: {right_speed:.1f}")

        # === Send to Arduino ===
        clamp_flag = 1 if box_height > 0.85 * max_height else 0
        command = f"{int(left_speed)},{int(right_speed)},{clamp_flag}\n"
        ser.write(command.encode('utf-8'))


        time.sleep(0.05) 

    except KeyboardInterrupt:
        print("Stopped")
        ser.close()


    except Exception as e:
        print(f"wrong:{e}")
