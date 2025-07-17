/*
  The following treats the Arduino as a mere dumb motor driver slave that simply complies with the commands received from the Pi
  The command handling has been kept pretty simple for testing purposes to get down the transmission and systems integration
  The previous code had a useless LCD attached that increased the command processing latency to 30 seconds because of lcd.print

  Created a separate 'timeout' for the turns because it turned too fast while using a common command timeout
  Added debug comments and functionality to toggle them on and off
  Autoclamp is handled by arduino and clamps when the object is within the range of the claw
  The camera needs to ping and send details once every 30 frames only, otherwise the hardware gets overwhelmed with the barrage of commands and we make no progress.

  These values have given the best performance till now. The drive and detection is smooth, even if the steps are slow.
  ======================= The PID needs to be tuned more before it's usable, it's speed corrections are horrendous rn =======================
*/
#include <AFMotor.h>
#include <Servo.h>
#include <NewPing.h>

// === Debug flag ===
#define DEBUG false   // <<<<< Toggle debug messages

Servo cameraServo;

// Initial angles for servos -- pls forgive the naming scheme i was using the servos for a rotating camera before
int cameraAngle = 0;

// === Motors ===
int DRIVE_SPEED_MAX = 200;
int DRIVE_SPEED_MID = 155;
int DRIVE_SPEED_MIN = 120;
int DRIVE_SPEED = DRIVE_SPEED_MAX;
const int MAX_SPEED = 230;
int TURN_SPEED1 = 150;
int TURN_SPEED0 = 118;

// === A timeout for turning ===
unsigned long turnStartTime = 0;
unsigned long turnDuration = 200;  // ms


AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);

// === Ultrasonic ===
#define TRIG_PIN 22
#define ECHO_PIN 23

// === State ===
bool clamped = false;
bool driving = false;
bool forward = true;
bool turningInPlace = false;
bool turnLeftInPlace = false;
bool turnRightInPlace = false;

unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 300;  // ms

// === Smoothed distance ===
float smoothedDistance = -1;
const float SMOOTH_ALPHA = 0.7;  // Low-pass filter factor

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

  float distance = readDistanceCM();
  if (distance > 0) {
    if (smoothedDistance < 0) smoothedDistance = distance;
    smoothedDistance = SMOOTH_ALPHA * smoothedDistance + (1 - SMOOTH_ALPHA) * distance;

    if (DEBUG) {
      Serial.print("Raw Distance: "); Serial.print(distance);
      Serial.print(" | Smoothed: "); Serial.println(smoothedDistance);
    }

    if (driving && forward) {
      if (smoothedDistance > 0 && smoothedDistance <= 5.0 && !clamped) {
        if (DEBUG) {
          Serial.print("Object close! Smoothed: ");
          Serial.println(smoothedDistance);
        }
        stopMotors();
        //delay(300);  // small pause for stability
        clamp();
      } else if (smoothedDistance > 45.0) {
        DRIVE_SPEED = DRIVE_SPEED_MAX;
      } else if (smoothedDistance <= 40.0 && smoothedDistance > 15.0) {
        DRIVE_SPEED = DRIVE_SPEED_MID;
      } else if (smoothedDistance <= 15.0 && smoothedDistance > 5.0) {
        DRIVE_SPEED = DRIVE_SPEED_MIN;
      } else {
        DRIVE_SPEED = DRIVE_SPEED_MAX; // fallback if distance is outside all ranges
      }
    } else {
      DRIVE_SPEED = DRIVE_SPEED_MAX; // if not driving forward, keep default
    }

  } else {
    // Ignore invalid reading; keep previous smoothedDistance
    if (DEBUG) Serial.println("Invalid distance (-1)");
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
      turnStartTime= millis();
      break;

    case 'D':
      driving = false;
      turningInPlace = true;
      turnRightInPlace = false;
      turnLeftInPlace = true;
      turnStartTime= millis();
      break;

    case 'X':
      driving = false;
      turningInPlace = false;
      stopMotors();
      break;

    case 'U':
      clamp();
      cameraAngle += 30;
      if (DEBUG) Serial.print("Camera angle: "), Serial.println(cameraAngle);
      break;

    case 'N':
      unclamp();
      cameraAngle -= 30;
      if (DEBUG) Serial.print("Camera angle: "), Serial.println(cameraAngle);
      break;
  }
}

void applyDrive() {
  int speed = DRIVE_SPEED;
  speed = constrain(speed, DRIVE_SPEED_MIN, DRIVE_SPEED_MAX);
  if (DEBUG) {
    Serial.print("Drive Speed: ");
    Serial.println(speed);
  }

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
  int speed = TURN_SPEED1;

  motor1.setSpeed(speed);
  motor2.setSpeed(speed);
  motor3.setSpeed(speed);
  motor4.setSpeed(speed);

  if (turnLeftInPlace) {
    motor1.run(BACKWARD);
    motor2.run(BACKWARD);
    motor3.run(FORWARD);
    motor4.run(FORWARD);
    //delay(200);
    //stopMotors();
  } else if (turnRightInPlace) {
    motor1.run(FORWARD);
    motor2.run(FORWARD);
    motor3.run(BACKWARD);
    motor4.run(BACKWARD);
    //delay(200);
    //stopMotors();
  }

  if (millis() - turnStartTime >= turnDuration) {
    turningInPlace = false;
    turnLeftInPlace = false;
    turnRightInPlace = false;
    stopMotors();
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

  long duration = pulseIn(ECHO_PIN, HIGH, 20000);
  if (duration == 0) return -1;
  return duration * 0.0343 / 2.0;
}

void clamp() {
  cameraServo.write(120);
  clamped = true;
  Serial.println("Clamped");
}

void unclamp() {
  cameraServo.write(180);
  clamped = false;
  Serial.println("Unclamped");
}
/*
  Note: The AFMotor library is used for motor control, and the Servo library is used for the camera servo.
  Ensure that the AFMotor library is compatible with your hardware setup.
*/
