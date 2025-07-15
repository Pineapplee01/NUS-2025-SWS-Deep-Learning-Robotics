/*
  The following treats the Arduino as a mere dumb motor driver slave that simply complies with the commands received from the Pi
  The command handling has been kept pretty simple for testing purposes to get down the transmission and systems integration
  The previous code had a useless LCD attached that increased the command processing latency to 30 seconds because of lcd.print

  The camera needs to ping and send details once every 30 frames only, otherwise the hardware gets overwhelmed with the barrage of commands and we make no progress.
  Now includes an auto clamp feature that clamps on an object when it is in range of the claw (after a delay of 1s)


  ======================= The PID needs to be tuned more before it's usable, it's speed corrections are horrendous rn =======================
*/
#include <AFMotor.h>
#include <Servo.h>
#include <NewPing.h>

Servo cameraServo;

// Initial angles for clamp servos -- pls forgive the naming scheme i was using the servos for a rotating camera before
int cameraAngle = 0;

// === Motors ===
#define DRIVE_SPEED 230
#define TURN_SPEED 100

AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);

// === Ultrasonic ===
#define TRIG_PIN 22
#define ECHO_PIN 23

// === Clamping State ===
bool clamped = false;

// === State ===
bool driving = false;
bool forward = true;
bool turningInPlace = false;
bool turnLeftInPlace = false;
bool turnRightInPlace = false;

unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 500;  // ms

void setup() {
  Serial.begin(115200);
  stopMotors();
  cameraServo.attach(46); //  servo connected to pin 46, pin 44 isn't working, 45 is spotty at best...
  cameraServo.write(cameraAngle);

  unclamp();

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  Serial.println("Bang-bang + ultrasonic ready!");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    char cmd = input.charAt(0);
    handleCommand(cmd);
    lastCommandTime = millis();
  }

  if (turningInPlace) {
    applyInPlaceTurn();
  } else if (driving) {
    applyDrive();
  } else {
    stopMotors();
  }

  // === Ultrasonic check -- will try to overide later ===
  float distance = readDistanceCM();

  if (distance > 0 && distance <= 6.0 && !clamped) {
    Serial.print("Object close! Distance: ");
    Serial.print(distance);
    Serial.println(" cm -> Clamping!");
    delay(1000);

    clamp();
  }

  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    driving = false;
    turningInPlace = false;
    stopMotors();
  }
}

void handleCommand(char cmd) {
  switch (cmd) {
    case 'W':
      driving = true;
      forward = true;
      turningInPlace = false;
      break;

    case 'S':
      driving = true;
      forward = false;
      turningInPlace = false;
      break;

    case 'A':
      driving = false;
      turningInPlace = true;
      turnLeftInPlace = false;
      turnRightInPlace = true;
      break;

    case 'D':
      driving = false;
      turningInPlace = true;
      turnRightInPlace = false;
      turnLeftInPlace = true;
      break;

    case 'X':
      driving = false;
      turningInPlace = false;
      stopMotors();
      break;

    case 'U': // Camera Tilt Up
      clamp();
      cameraAngle+=30;
      Serial.print("Camera angle: "); Serial.println(cameraAngle);
      break;

    case 'N': // Camera Tilt Down
      unclamp();
      cameraAngle+30;
      Serial.print("Camera angle: "); Serial.println(cameraAngle);
      break;  
  }
}

void applyDrive() {
  int speed = DRIVE_SPEED;

  motor1.setSpeed(speed);
  motor2.setSpeed(speed);
  motor3.setSpeed(speed);
  motor4.setSpeed(speed);

  if (forward) {
    motor1.run(FORWARD);
    motor2.run(FORWARD);
    motor3.run(FORWARD);
    motor4.run(FORWARD);
  } else {
    motor1.run(BACKWARD);
    motor2.run(BACKWARD);
    motor3.run(BACKWARD);
    motor4.run(BACKWARD);
  }
}

void applyInPlaceTurn() {
  int speed = TURN_SPEED;

  motor1.setSpeed(speed);
  motor2.setSpeed(speed);
  motor3.setSpeed(speed);
  motor4.setSpeed(speed);

  if (turnLeftInPlace) {
    motor1.run(BACKWARD);
    motor2.run(BACKWARD);
    motor3.run(FORWARD);
    motor4.run(FORWARD);
  } else if (turnRightInPlace) {
    motor1.run(FORWARD);
    motor2.run(FORWARD);
    motor3.run(BACKWARD);
    motor4.run(BACKWARD);
  }
}

void stopMotors() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
}

float readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 20000); // 20 ms timeout

  if (duration == 0) return -1;

  float distance = duration * 0.0343 / 2.0;
  return distance;
}

void clamp() {
  cameraServo.write(120); 
  clamped = true;
  Serial.println("Clamped");
}

void unclamp() {
  cameraServo.write(170); // Adjust angle to match your claw open position
  clamped = false;
  Serial.println("Unclamped");
}