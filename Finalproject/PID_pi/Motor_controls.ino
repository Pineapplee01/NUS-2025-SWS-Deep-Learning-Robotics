#include <AFMotor.h>
#include <Arduino.h>

AF_DCMotor rightMotorback(1);
AF_DCMotor rightMotorfront(2);
AF_DCMotor leftMotorback(3);
AF_DCMotor leftMotorfront(4);

/*
const byte BUFFER_SIZE = 16;
char buffer[BUFFER_SIZE];
byte bufferIndex = 0;
*/

void setup() {
  Serial.begin(115200);
  Serial.println("Robot activating...");
  delay(500);
  Serial.println("Robot ready!");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); //string parse seems fast enough to not warrant a buffer method
    int commaIndex = input.indexOf(',');
    if (commaIndex > 0) {
      int leftSpeed = input.substring(0, commaIndex).toInt();
      int rightSpeed = input.substring(commaIndex + 1).toInt();

      Serial.print("Left wheel speed: "); Serial.print(leftSpeed); Serial.print(" | Right wheel speed: "); Serial.println(rightSpeed);

      move(leftSpeed, rightSpeed);
    }
  }    
}

void move(int leftSpeed, int rightSpeed) {
  leftMotorback.setSpeed(leftSpeed);
  leftMotorfront.setSpeed(leftSpeed);
  rightMotorback.setSpeed(rightSpeed);
  rightMotorfront.setSpeed(rightSpeed);

  leftMotorfront.run(FORWARD);
  leftMotorback.run(FORWARD);
  rightMotorfront.run(FORWARD);
  rightMotorback.run(FORWARD);
}

/*
void parseCommand(char* cmd) {
  // Expecting: "120,80"
  int commaIndex = -1;
  for (byte i = 0; i < BUFFER_SIZE; i++) {
    if (cmd[i] == ',') {
      commaIndex = i;
      break;
    }
  }

  if (commaIndex > 0) {
    cmd[commaIndex] = '\0';  // Split into two substrings
    int leftSpeed = atoi(cmd);
    int rightSpeed = atoi(cmd + commaIndex + 1);

    leftSpeed = constrain(leftSpeed, 0, 255);
    rightSpeed = constrain(rightSpeed, 0, 255);

    //still using tank style turn
  }
}      

*/