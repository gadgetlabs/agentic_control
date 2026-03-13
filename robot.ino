#include <CRSFforArduino.hpp>
#include <Wire.h>
#include <PID_v1.h>
#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

// IMU & Compass libraries
#include "bmm150.h"
#include "bmm150_defs.h"
#include <SparkFunLSM6DS3.h>        // Grove 6-axis IMU (LSM6DS3 based)

/* ===================== CRSF/ELRS ===================== */
CRSFforArduino crsf = CRSFforArduino(&Serial3);

/* ===================== IMU & COMPASS ===================== */
BMM150 compass;
LSM6DS3 imu(I2C_MODE, 0x6A);

/* ===================== CONSTANTS ===================== */
constexpr float ENCODER_CPR    = 542.7;  // FIT0186: 16 * 33.875 gear ratio
constexpr float MAX_RPM        = 251.0;
constexpr float LOOP_DT        = 0.01;   // 100 Hz

constexpr float WHEEL_DIAMETER = 0.065;
constexpr float WHEEL_BASE     = 0.20;

constexpr uint32_t CMD_TIMEOUT_MS = 200;

constexpr int CRSF_CENTER = 1500;
constexpr int CRSF_RANGE  = 500;

/* ===================== PINS ===================== */
// Motor driver (Cytron MDD10A)
#define LEFT_PWM   5
#define LEFT_DIR   4
#define RIGHT_PWM  6
#define RIGHT_DIR  7

// Encoders (FIT0186)
#define LEFT_ENC_A  2
#define LEFT_ENC_B  3
#define RIGHT_ENC_A 8
#define RIGHT_ENC_B 9

// RGB LED Stick (Grove 10x WS2813 Mini)
#define LED_STICK_PIN 11
#define LED_COUNT     10

// Lidar (LDS02RR) - MOSFET controls power to lidar + motor
#define LIDAR_MOSFET 10

// --- COMMENTED OUT: Button not wired ---
// #define BUTTON_PIN 12

#define LED_PIN    13

/* ===================== LDS02RR LIDAR ===================== */
constexpr int LIDAR_PACKET_SIZE = 22;
constexpr uint8_t LIDAR_START_BYTE = 0xFA;

uint8_t lidarBuf[LIDAR_PACKET_SIZE];
uint8_t lidarBufIdx = 0;
uint16_t lidarDistances[360];
uint16_t lidarRpm = 0;
bool lidarNewScan = false;

/* ===================== RGB LED STICK ===================== */
Adafruit_NeoPixel strip(LED_COUNT, LED_STICK_PIN, NEO_GRB + NEO_KHZ800);

enum Emotion { EMO_IDLE = 0, EMO_HAPPY, EMO_THINKING, EMO_SAD, EMO_ANGRY };
Emotion currentEmotion = EMO_IDLE;
uint32_t emoLastUpdate = 0;
uint16_t emoStep = 0;

/* ===================== ENCODERS ===================== */
volatile int32_t leftTicks  = 0;
volatile int32_t rightTicks = 0;
volatile bool leftDir  = true;
volatile bool rightDir = true;
volatile byte leftLastA  = LOW;
volatile byte rightLastA = LOW;

void leftEncISR() {
  byte stateA = digitalRead(LEFT_ENC_A);
  if (leftLastA == LOW && stateA == HIGH) {
    byte stateB = digitalRead(LEFT_ENC_B);
    if (stateB == LOW && leftDir) leftDir = false;
    else if (stateB == HIGH && !leftDir) leftDir = true;
  }
  leftLastA = stateA;
  if (leftDir) leftTicks++; else leftTicks--;
}

void rightEncISR() {
  byte stateA = digitalRead(RIGHT_ENC_A);
  if (rightLastA == LOW && stateA == HIGH) {
    byte stateB = digitalRead(RIGHT_ENC_B);
    if (stateB == LOW && rightDir) rightDir = false;
    else if (stateB == HIGH && !rightDir) rightDir = true;
  }
  rightLastA = stateA;
  if (rightDir) rightTicks++; else rightTicks--;
}

/* ===================== PID ===================== */
double leftRPM, leftTargetRPM, leftOut;
double rightRPM, rightTargetRPM, rightOut;
PID leftPID(&leftRPM, &leftOut, &leftTargetRPM, 0.02, 0.2, 0.0005, DIRECT);
PID rightPID(&rightRPM, &rightOut, &rightTargetRPM, 0.02, 0.2, 0.0005, DIRECT);
int32_t lastLeftTicks  = 0;
int32_t lastRightTicks = 0;

/* ===================== STATE ===================== */
uint32_t lastLoop = 0;
uint32_t lastCmd  = 0;
uint32_t lastRc   = 0;
uint32_t lastSerialOut = 0;
float prevThrottle = 0.0f;
float prevSteering = 0.0f;

enum Mode { MODE_RC, MODE_AUTONOMOUS };
Mode currentMode = MODE_AUTONOMOUS;

uint32_t lastBlink = 0;
bool ledState = false;
uint32_t blinkInterval = 500;

bool lidarEnabled = false;

float rcThrottle = 0.0f;
float rcSteering = 0.0f;
bool armed = false;

bool compassOk = false;
bool imuOk = false;

/* ===================== SERIAL COMMAND PARSER ===================== */
char   _cmdBuf[64];
uint8_t _cmdBufIdx   = 0;
float  _cmdThrottle  = 0.0f;
float  _cmdSteering  = 0.0f;

void parseCommand(const char* line) {
  char buf[64];
  strncpy(buf, line, 63);
  buf[63] = 0;
  char* tok = strtok(buf, ",");
  if (!tok || strcmp(tok, "CMD") != 0) return;
  tok = strtok(NULL, ",");
  if (!tok) return;

  if (strcmp(tok, "DRIVE") == 0) {
    char* t_s = strtok(NULL, ",");
    char* s_s = strtok(NULL, ",");
    if (t_s && s_s) {
      _cmdThrottle = constrain(atoi(t_s) / 100.0f, -1.0f, 1.0f);
      _cmdSteering = constrain(atoi(s_s) / 100.0f, -1.0f, 1.0f);
      lastCmd = millis();
    }
  } else if (strcmp(tok, "EMOTION") == 0) {
    char* e_s = strtok(NULL, ",");
    if (e_s) {
      uint8_t emo = (uint8_t)atoi(e_s);
      if (emo <= EMO_ANGRY) { currentEmotion = (Emotion)emo; emoStep = 0; }
    }
  }
}

void readSerialCommands() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      _cmdBuf[_cmdBufIdx] = 0;
      parseCommand(_cmdBuf);
      _cmdBufIdx = 0;
    } else if (c != '\r' && _cmdBufIdx < 63) {
      _cmdBuf[_cmdBufIdx++] = c;
    }
  }
}
float compassX = 0, compassY = 0, compassZ = 0;
float accelX, accelY, accelZ;
float gyroX, gyroY, gyroZ;

/* ===================== UTILS ===================== */
float ticksToRPM(int32_t deltaTicks) {
  return (deltaTicks / ENCODER_CPR / LOOP_DT) * 60.0;
}

void stopMotors() {
  analogWrite(LEFT_PWM, 0);
  analogWrite(RIGHT_PWM, 0);
  digitalWrite(LEFT_DIR, LOW);
  digitalWrite(RIGHT_DIR, LOW);
  leftTargetRPM = 0;
  rightTargetRPM = 0;
}

void driveMotor(int pwmPin, int dirPin, float v) {
  v = constrain(v, -1.0f, 1.0f);
  digitalWrite(dirPin, v >= 0 ? HIGH : LOW);
  analogWrite(pwmPin, (int)(abs(v) * 255.0f));
}

float applyDeadband(float v, float db = 0.08f) {
  if (abs(v) < db) return 0.0f;
  return (v > 0) ? (v - db) / (1.0f - db) : (v + db) / (1.0f - db);
}

float ramp(float current, float target, float step) {
  if (target > current + step) return current + step;
  if (target < current - step) return current - step;
  return target;
}

float crsfToNorm(int val) {
  return constrain((val - CRSF_CENTER) / (float)CRSF_RANGE, -1.0f, 1.0f);
}

/* ===================== MIXER ===================== */
void mixSkidSteer(float throttle, float steering) {
  steering *= (1.0f - 0.5f * abs(throttle));
  float left  = throttle + steering;
  float right = throttle - steering;
  float maxMag = max(abs(left), abs(right));
  if (maxMag > 1.0f) { left /= maxMag; right /= maxMag; }
  leftTargetRPM  = left  * MAX_RPM;
  rightTargetRPM = right * MAX_RPM;
}

/* ===================== LED ===================== */
void updateLED() {
  uint32_t now = millis();
  if (now - lastBlink >= blinkInterval) {
    lastBlink = now;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }
}

void setBlinkRate(uint32_t interval) { blinkInterval = interval; }

/* ===================== EMOTION LED ANIMATIONS ===================== */
void updateEmotion() {
  uint32_t now = millis();

  switch (currentEmotion) {
    case EMO_IDLE: {
      // Dim white gentle breathe (2s cycle)
      if (now - emoLastUpdate < 20) return;
      emoLastUpdate = now;
      emoStep = (emoStep + 1) % 200;
      // Triangle wave: 0→100→0 over 200 steps (4s at 50Hz)
      uint8_t bright = emoStep < 100 ? emoStep : 200 - emoStep;
      bright = 5 + (bright * 30) / 100;  // range 5..35
      for (int i = 0; i < LED_COUNT; i++) {
        strip.setPixelColor(i, strip.Color(bright, bright, bright));
      }
      break;
    }
    case EMO_HAPPY: {
      // Green with yellow accents, scanning back and forth
      if (now - emoLastUpdate < 60) return;
      emoLastUpdate = now;
      emoStep = (emoStep + 1) % (LED_COUNT * 2);
      // Ping-pong position
      int pos = emoStep < LED_COUNT ? emoStep : (LED_COUNT * 2 - 1) - emoStep;
      for (int i = 0; i < LED_COUNT; i++) {
        int dist = abs(i - pos);
        if (dist == 0) {
          strip.setPixelColor(i, strip.Color(180, 255, 0));   // bright yellow-green
        } else if (dist == 1) {
          strip.setPixelColor(i, strip.Color(40, 180, 0));    // medium green
        } else {
          strip.setPixelColor(i, strip.Color(0, 30, 0));      // dim green
        }
      }
      break;
    }
    case EMO_THINKING: {
      // Blue chase/spinner cycling around
      if (now - emoLastUpdate < 80) return;
      emoLastUpdate = now;
      emoStep = (emoStep + 1) % LED_COUNT;
      for (int i = 0; i < LED_COUNT; i++) {
        int dist = (LED_COUNT + i - (int)emoStep) % LED_COUNT;
        if (dist == 0) {
          strip.setPixelColor(i, strip.Color(0, 80, 255));    // bright blue
        } else if (dist == 1) {
          strip.setPixelColor(i, strip.Color(0, 30, 120));    // medium blue
        } else if (dist == 2) {
          strip.setPixelColor(i, strip.Color(0, 10, 40));     // dim blue
        } else {
          strip.setPixelColor(i, strip.Color(0, 0, 5));       // near off
        }
      }
      break;
    }
    case EMO_SAD: {
      // Deep blue, slow dim breathing (3s cycle)
      if (now - emoLastUpdate < 30) return;
      emoLastUpdate = now;
      emoStep = (emoStep + 1) % 200;
      uint8_t bright = emoStep < 100 ? emoStep : 200 - emoStep;
      bright = 2 + (bright * 25) / 100;  // range 2..27
      for (int i = 0; i < LED_COUNT; i++) {
        strip.setPixelColor(i, strip.Color(0, 0, bright));
      }
      break;
    }
    case EMO_ANGRY: {
      // Red, fast alternating flash
      if (now - emoLastUpdate < 80) return;
      emoLastUpdate = now;
      emoStep = (emoStep + 1) % 4;
      for (int i = 0; i < LED_COUNT; i++) {
        bool on = (emoStep < 2) ? (i % 2 == 0) : (i % 2 == 1);
        strip.setPixelColor(i, on ? strip.Color(255, 0, 0) : strip.Color(0, 0, 0));
      }
      break;
    }
  }
  strip.show();
}

/* ===================== LIDAR CONTROL ===================== */
void lidarOn() {
  digitalWrite(LIDAR_MOSFET, HIGH);
  lidarEnabled = true;
  memset(lidarDistances, 0, sizeof(lidarDistances));
  Serial.println("[MOSFET] Pin 10 HIGH - Lidar ON");
}

void lidarOff() {
  digitalWrite(LIDAR_MOSFET, LOW);
  lidarEnabled = false;
  Serial.println("[MOSFET] Pin 10 LOW - Lidar OFF");
}

/* ===================== LDS02RR PARSING ===================== */
void processLidarPacket() {
  if (lidarBuf[0] != LIDAR_START_BYTE) return;
  uint8_t index = lidarBuf[1] - 0xA0;
  if (index > 89) return;
  lidarRpm = (lidarBuf[3] << 8 | lidarBuf[2]) / 64;
  if (index == 0) lidarNewScan = true;
  for (int i = 0; i < 4; i++) {
    int offset = 4 + i * 4;
    uint16_t dist = lidarBuf[offset] | (lidarBuf[offset + 1] << 8);
    bool invalid = dist & 0x8000;
    dist &= 0x3FFF;
    int angle = index * 4 + i;
    if (angle < 360) lidarDistances[angle] = invalid ? 0 : dist;
  }
}

void readLidar() {
  if (!lidarEnabled) return;
  while (Serial1.available()) {
    uint8_t b = Serial1.read();
    if (lidarBufIdx == 0 && b != LIDAR_START_BYTE) continue;
    lidarBuf[lidarBufIdx++] = b;
    if (lidarBufIdx >= LIDAR_PACKET_SIZE) {
      processLidarPacket();
      lidarBufIdx = 0;
    }
  }
}

/* ===================== I2C SCANNER ===================== */
void scanI2C() {
  Serial.println("I2C bus scan:");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  Found device at 0x");
      if (addr < 16) Serial.print("0");
      Serial.print(addr, HEX);
      if (addr == 0x10) Serial.print(" (BMM150 compass)");
      else if (addr == 0x13) Serial.print(" (BMM150 alt addr)");
      else if (addr == 0x6A) Serial.print(" (LSM6DS3 IMU)");
      else if (addr == 0x6B) Serial.print(" (LSM6DS3 alt addr)");
      Serial.println();
      found++;
    }
  }
  Serial.print("  Total: ");
  Serial.print(found);
  Serial.println(" device(s)");
}

/* ===================== SENSOR READING ===================== */
void readSensors() {
  if (compassOk) {
    compass.read_mag_data();
    compassX = compass.raw_mag_data.raw_datax;
    compassY = compass.raw_mag_data.raw_datay;
    compassZ = compass.raw_mag_data.raw_dataz;
  }
  if (imuOk) {
    accelX = imu.readFloatAccelX();
    accelY = imu.readFloatAccelY();
    accelZ = imu.readFloatAccelZ();
    gyroX = imu.readFloatGyroX();
    gyroY = imu.readFloatGyroY();
    gyroZ = imu.readFloatGyroZ();
  }
}

/* ===================== SERIAL SENSOR OUTPUT ===================== */
void serialSensorOutput() {
  Serial.print("$IMU,");
  Serial.print(accelX, 3); Serial.print(",");
  Serial.print(accelY, 3); Serial.print(",");
  Serial.print(accelZ, 3); Serial.print(",");
  Serial.print(gyroX, 2); Serial.print(",");
  Serial.print(gyroY, 2); Serial.print(",");
  Serial.println(gyroZ, 2);

  Serial.print("$CMP,");
  Serial.print(compassX, 1); Serial.print(",");
  Serial.print(compassY, 1); Serial.print(",");
  Serial.println(compassZ, 1);

  // Real odometry from encoders
  float leftMps  = (leftRPM  / 60.0) * PI * WHEEL_DIAMETER;
  float rightMps = (rightRPM / 60.0) * PI * WHEEL_DIAMETER;
  float linear  = (leftMps + rightMps) / 2.0;
  float angular = (rightMps - leftMps) / WHEEL_BASE;

  Serial.print("$ODO,");
  Serial.print(linear, 3); Serial.print(",");
  Serial.println(angular, 3);

  Serial.print("$RPM,");
  Serial.print(leftRPM, 1); Serial.print(",");
  Serial.println(rightRPM, 1);

  if (lidarEnabled && lidarNewScan) {
    Serial.print("$LDR,");
    Serial.print(lidarRpm);
    for (int i = 0; i < 360; i++) {
      Serial.print(",");
      Serial.print(lidarDistances[i]);
    }
    Serial.println();
    lidarNewScan = false;
  }
}

/* ===================== SETUP ===================== */
void setup() {
  // CRITICAL: Set motor pins LOW before configuring as outputs
  digitalWrite(LEFT_PWM, LOW);
  digitalWrite(LEFT_DIR, LOW);
  digitalWrite(RIGHT_PWM, LOW);
  digitalWrite(RIGHT_DIR, LOW);
  pinMode(LEFT_PWM, OUTPUT);  pinMode(LEFT_DIR, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT); pinMode(RIGHT_DIR, OUTPUT);
  stopMotors();

  Serial.begin(115200);
  delay(100);
  Serial.println("CHAOS HAL Starting...");
  Serial.println(">>> DISARMED - Flip CH5 to arm <<<");

  // Encoder pins
  pinMode(LEFT_ENC_A, INPUT_PULLUP);  pinMode(LEFT_ENC_B, INPUT_PULLUP);
  pinMode(RIGHT_ENC_A, INPUT_PULLUP); pinMode(RIGHT_ENC_B, INPUT_PULLUP);

  // --- COMMENTED OUT: Button not wired ---
  // pinMode(BUTTON_PIN, INPUT_PULLUP);

  pinMode(LED_PIN, OUTPUT);

  // RGB LED Stick
  strip.begin();
  strip.show();  // all off

  // Lidar MOSFET
  pinMode(LIDAR_MOSFET, OUTPUT);
  lidarOff();

  // Lidar UART
  Serial1.begin(115200);

  // I2C for compass and IMU
  Wire.begin();
  Wire.setClock(100000);  // 100kHz - safe for Grove modules
  delay(200);

  scanI2C();

  Serial.print("Initializing compass (BMM150 @ 0x10)... ");
  int bmm_result = compass.initialize();
  if (bmm_result == BMM150_E_ID_NOT_CONFORM) {
    Serial.println("FAILED (chip ID mismatch)");
  } else if (bmm_result != BMM150_OK) {
    Serial.print("FAILED (code=");
    Serial.print(bmm_result);
    Serial.println(")");
  } else {
    Serial.println("OK");
    compassOk = true;
  }

  Serial.print("Initializing IMU (LSM6DS3 @ 0x6A)... ");
  int imu_result = imu.begin();
  if (imu_result != 0) {
    Serial.print("FAILED (code=");
    Serial.print(imu_result);
    Serial.println(")");
  } else {
    Serial.println("OK");
    imuOk = true;
  }

  Serial.print("SENSOR STATUS: compass=");
  Serial.print(compassOk ? "OK" : "FAIL");
  Serial.print(" imu=");
  Serial.println(imuOk ? "OK" : "FAIL");

  // ELRS/CRSF on Serial3
  if (crsf.begin()) {
    Serial.println("CRSF initialized on Serial3");
  } else {
    Serial.println("CRSF init failed!");
  }

  // Encoder interrupts
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_A), leftEncISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_A), rightEncISR, CHANGE);

  // PID
  leftPID.SetOutputLimits(-1.0, 1.0);  leftPID.SetMode(AUTOMATIC);
  rightPID.SetOutputLimits(-1.0, 1.0); rightPID.SetMode(AUTOMATIC);

  lastLoop = lastCmd = lastRc = millis();
  setBlinkRate(500);

  lidarOn();
  Serial.println("Ready! Streaming sensors. Flip CH5 to arm for RC control.");
}

/* ===================== LOOP ===================== */
void loop() {
  uint32_t now = millis();

  crsf.update();
  updateLED();
  updateEmotion();
  readLidar();

  // Main control loop at 100Hz
  if (now - lastLoop < LOOP_DT * 1000) return;
  lastLoop += LOOP_DT * 1000;

  float throttle = 0.0f;
  float steering = 0.0f;

  // Read RC channels
  int ch1_us = crsf.rcToUs(crsf.getChannel(1));
  int ch3_us = crsf.rcToUs(crsf.getChannel(3));
  int ch5_us = crsf.rcToUs(crsf.getChannel(5));

  bool validRc = (ch1_us >= 900 && ch1_us <= 2100 && ch3_us >= 900 && ch3_us <= 2100);

  // Arming: CH5 > 1700 = armed (RC mode), else autonomous mode
  bool prevArmed = armed;
  armed = validRc && (ch5_us > 1700);

  // Mode changes
  if (armed != prevArmed) {
    if (armed) {
      currentMode = MODE_RC;
      lidarOff();
      Serial.println(">>> ARMED (RC MODE) <<<");
    } else {
      currentMode = MODE_AUTONOMOUS;
      lidarOn();
      stopMotors();
      Serial.println(">>> DISARMED (AUTONOMOUS MODE) - Sensors streaming <<<");
    }
  }

  if (currentMode == MODE_RC) {
    if (validRc && armed) {
      rcSteering = crsfToNorm(ch1_us);
      rcThrottle = crsfToNorm(ch3_us);
      throttle = rcThrottle;
      steering = rcSteering;
      lastRc = now;
      setBlinkRate(100);
    } else if (now - lastRc > CMD_TIMEOUT_MS) {
      stopMotors();
      setBlinkRate(1000);
    }
  } else {
    // Autonomous Mode: serial commands from Jetson over USB
    readSerialCommands();
    if (now - lastCmd > CMD_TIMEOUT_MS) {
      _cmdThrottle = 0.0f;
      _cmdSteering = 0.0f;
    }
    throttle = _cmdThrottle;
    steering = _cmdSteering;
    setBlinkRate(500);
  }

  // Apply controls
  throttle = applyDeadband(throttle);
  steering = applyDeadband(steering);
  throttle = ramp(prevThrottle, throttle, 0.05f);
  steering = ramp(prevSteering, steering, 0.05f);
  prevThrottle = throttle;
  prevSteering = steering;

  // Mixer sets target RPMs
  mixSkidSteer(throttle, steering);

  // Measure actual RPM from encoders
  int32_t lt = leftTicks, rt = rightTicks;
  leftRPM  = ticksToRPM(lt - lastLeftTicks);
  rightRPM = ticksToRPM(rt - lastRightTicks);
  lastLeftTicks = lt;
  lastRightTicks = rt;

  // PID computes motor output
  leftPID.Compute();
  rightPID.Compute();
  driveMotor(LEFT_PWM, LEFT_DIR, leftOut);
  driveMotor(RIGHT_PWM, RIGHT_DIR, rightOut);

  // Read sensors
  readSensors();

  // Odometry from encoders
  float leftMps  = (leftRPM  / 60.0) * PI * WHEEL_DIAMETER;
  float rightMps = (rightRPM / 60.0) * PI * WHEEL_DIAMETER;
  float linear  = (leftMps + rightMps) / 2.0;
  float angular = (rightMps - leftMps) / WHEEL_BASE;

  // Serial sensor output in autonomous mode (10Hz)
  if (currentMode == MODE_AUTONOMOUS && now - lastSerialOut >= 100) {
    lastSerialOut = now;
    serialSensorOutput();
  }
}
