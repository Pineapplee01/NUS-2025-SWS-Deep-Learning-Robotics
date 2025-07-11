#include <AFMotor.h>
#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);  // I2C address: 0x27, checked with Google's code

Servo cameraServo;

int cameraAngle = 0;  // Start at center
// Be warned, the LCD commands take time to process

// === Motors ===
#define DRIVE_SPEED 230

#define TURN_SPEED 180 
// I think this turn speed is still too fast to achieve the precision we need, especially with web based controls

AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);

// === State ===
bool driving = false;
bool forward = true;
bool steeringLeft = false;
bool steeringRight = false;

bool turningInPlace = false;
bool turnLeftInPlace = false;
bool turnRightInPlace = false;

int steerOffset = 0; // -64 (left) to +64 (right)

unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 300;

// === The active speed ===

void setup() {
  Serial.begin(115200);
  stopMotors();
  cameraServo.attach(46); // Connect to pin 46
  cameraServo.write(cameraAngle);
  lcd.init();           // Initialize LCD
  lcd.backlight();      // Turn on backlight
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Robot Ready!");

}

void loop() {
  while (Serial.available()) {
    char cmd = Serial.read();
    handleCommand(cmd);
    lastCommandTime = millis();
  }

  // === Ramp steering ===
  if (steeringLeft) {
    steerOffset -= 1;
    if (steerOffset < -64) steerOffset = -64;
  } else if (steeringRight) {
    steerOffset += 1;
    if (steerOffset > 64) steerOffset = 64;
  } else {
    if (steerOffset > 0) {
      steerOffset -= 1;
      if (steerOffset < 0) steerOffset = 0;
    } else if (steerOffset < 0) {
      steerOffset += 1;
      if (steerOffset > 0) steerOffset = 0;
    }
  }

  // === Drive ===
  if (turningInPlace) {
    applyInPlaceTurn();
  } else if (driving) {
    applyDrive();
  } else {
    stopMotors();
  }

  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    driving = false;
    turningInPlace = false;
  }

  delay(10);
}

void handleCommand(char cmd) {
  lcd.setCursor(0,0);
  lcd.print("Navigating...");
  lcd.setCursor(0, 1);
  lcd.print("Cmd: ");
  lcd.print(cmd);
  lcd.print("    ");

  switch (cmd) {
    case 'W':
      driving = true;
      forward = true;
      break;

    case 'S':
      driving = true;
      forward = false;
      break;

    case 'A':
      if (!driving) {
        turningInPlace = true;
        turnLeftInPlace = false;
        turnRightInPlace = true;
      } else {
        steeringLeft = false;
        steeringRight = true;
      }
      break;

    case 'D':
      if (!driving) {
        turningInPlace = true;
        turnRightInPlace = false;
        turnLeftInPlace = true;
      } else {
        steeringRight = false; 
        steeringLeft = true;
      }
      break;

    case 'X':
      driving = false;
      turningInPlace = false;
      steeringLeft = false;
      steeringRight = false;
      break;

    case 'U':  // Tilt up
      cameraAngle += 30;
      if (cameraAngle > 180) cameraAngle = 180;
      cameraServo.write(cameraAngle);
      Serial.print("Camera angle: "); Serial.println(cameraAngle);
      break;

    case 'N':  // Tilt down
      cameraAngle -= 30;
      if (cameraAngle < 0) cameraAngle = 0;
      cameraServo.write(cameraAngle);
      Serial.print("Camera angle: "); Serial.println(cameraAngle);
      break;  

  }
  lcd.setCursor(0,0);
  lcd.print("What next?");
};

void applyDrive() {
  int temp = (DRIVE_SPEED * steerOffset);
  int steerEffect = temp / 128;
  int leftSpeed = constrain(DRIVE_SPEED - steerEffect, 0, 255);
  int rightSpeed = constrain(DRIVE_SPEED + steerEffect, 0, 255);

  motor1.setSpeed(leftSpeed);
  motor2.setSpeed(leftSpeed);
  motor3.setSpeed(rightSpeed);
  motor4.setSpeed(rightSpeed);

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

  Serial.print("L: "); Serial.print(leftSpeed);
  Serial.print(" | R: "); Serial.print(rightSpeed);
  Serial.print(" | Offset: "); Serial.print(steerOffset);
  Serial.print(" | Speed: "); Serial.println(DRIVE_SPEED);
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

  Serial.print("In-place turning | Speed: ");
  Serial.println(TURN_SPEED);
}

void stopMotors() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
}
