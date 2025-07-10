import time
import serial

class PiD:
    #constructor
    def __init__(self, Kp, Ki, Kd, setpoint=0):
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
heading_pid = PiD(Kp=0.01, Ki=0.0, Kd=0.002)  # Tuning required
speed_pid = PiD(Kp=1.5, Ki=0.1, Kd=0.2)  # Tuning required

# === Robot parameters ===
base_speed = 170 #normal drive  
max_turn = 100   

last_time = time.time()

try:
    while True:
        # === Get error ===
        # Get camera input here: ask Hammer
        heading_error = 0 #<-- replace with whatever calculated above (box offset)
        speed_error = 0 #<-- replace with whatever calculated from ablove (box size)

        # === Time loop update ===
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        turn_correction = heading_pid.update(heading_error, dt)
        speed_correction = speed_pid.update(speed_error, dt)

        # Limit turn output
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
        command = f"{int(left_speed)},{int(right_speed)}\n"
        ser.write(command.encode('utf-8'))

        time.sleep(0.05) 

except KeyboardInterrupt:
    print("Stopped")
    ser.close()
