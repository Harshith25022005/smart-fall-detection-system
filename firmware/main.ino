#include <Wire.h>
#include <BluetoothSerial.h>
#include <pgmspace.h>

#include <TensorFlowLite_ESP32.h>
#include "model.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// =====================================================
// BLUETOOTH
// BT transmits ONLY on:
//   1. Fall detected  — automatic push
//   2. 's' received   — manual query reply
// No streaming. No periodic sends.
// Keeping BT radio idle = no I2C corruption.
// =====================================================

BluetoothSerial SerialBT;

// =====================================================
// MPU6050
// =====================================================

#define MPU_ADDR 0x68

// =====================================================
// MODEL CONFIG
// =====================================================

#define WINDOW_SIZE  50
#define CHANNELS      4
#define NUM_CLASSES   5

// =====================================================
// NORMALIZATION
// =====================================================

float mean[4] = {
  0.73035283,
  6.94289265,
  0.39518895,
  12.00877499
};

float stdv[4] = {
  7.21604568,
  7.09905880,
  5.04718346,
  5.71180210
};

// =====================================================
// TFLITE
// =====================================================

const tflite::Model* tflModel =
    tflite::GetModel(model);

tflite::MicroErrorReporter micro_error_reporter;

tflite::MicroMutableOpResolver<11> resolver;

constexpr int tensor_arena_size = 40 * 1024;
uint8_t tensor_arena[tensor_arena_size];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// =====================================================
// LABELS
// =====================================================

const char* labels[] = {
  "Falling",
  "Jogging",
  "Sitting",
  "Standing",
  "Walking"
};

// =====================================================
// SENSOR WINDOW
// =====================================================

float sensorWindow[WINDOW_SIZE][CHANNELS];
int   realSampleCount = 0;

// =====================================================
// TIMING
// =====================================================

unsigned long lastSampleTime = 0;
const int sampleInterval = 50;   // 20 Hz

// =====================================================
// NON-BLOCKING LED
// =====================================================

bool          ledBlinking    = false;
int           ledTogglesDone = 0;
int           ledTogglesMax  = 0;
unsigned long ledLastToggle  = 0;
const int     ledToggleMs    = 150;
bool          ledState       = false;

void startBlink(int blinks) {
  ledBlinking    = true;
  ledTogglesDone = 0;
  ledTogglesMax  = blinks * 2;
  ledLastToggle  = millis();
  ledState       = false;
}

void updateLED() {
  if (!ledBlinking) return;
  if (millis() - ledLastToggle < ledToggleMs) return;
  ledState = !ledState;
  digitalWrite(2, ledState ? HIGH : LOW);   // onboard LED pin 2
  ledLastToggle = millis();
  ledTogglesDone++;
  if (ledTogglesDone >= ledTogglesMax) {
    ledBlinking = false;
    digitalWrite(2, LOW);
  }
}

// =====================================================
// FALL TIMERS
// =====================================================

unsigned long lastFallTime = 0;
const unsigned long fallCooldown    = 5000;
const unsigned long fallDisplayTime = 3000;

// =====================================================
// ACTIVITY SMOOTHING
//
// FIX: Confidence fallback uses a VOTE BUFFER,
// not lastPrediction or currentActivity.
// This breaks the Jogging lock-in completely.
// =====================================================

String currentActivity = "Unknown";

// Vote buffer: last N raw predictions from model
#define VOTE_SIZE 5
String voteBuffer[VOTE_SIZE];
int    voteIdx = 0;

void initVoteBuffer(String initial) {
  for (int i = 0; i < VOTE_SIZE; i++)
    voteBuffer[i] = initial;
}

// Majority vote across the buffer
String getMajority() {
  // Count each label
  const char* candidates[] = {
    "Jogging", "Sitting", "Standing", "Walking"
  };
  int counts[4] = {0, 0, 0, 0};
  for (int i = 0; i < VOTE_SIZE; i++) {
    for (int c = 0; c < 4; c++) {
      if (voteBuffer[i] == candidates[c]) {
        counts[c]++;
        break;
      }
    }
  }
  int best = 0;
  for (int c = 1; c < 4; c++) {
    if (counts[c] > counts[best])
      best = c;
  }
  return String(candidates[best]);
}

// =====================================================
// FALL FSM
// =====================================================

#define IMPACT_THRESH       20.0f
#define POST_REST_MIN        7.0f
#define POST_REST_MAX       14.0f
#define POST_REST_WINDOW_MS 2000
#define POST_REST_HOLD_MS    500
#define PRE_IMPACT_MAX      16.5f
#define PRE_BUF_SIZE          10

float preBuf[PRE_BUF_SIZE];
int   preBufIdx  = 0;
bool  preBufFull = false;

enum FallPhase { PHASE_IDLE, PHASE_IMPACT };
FallPhase     fallPhase = PHASE_IDLE;
unsigned long phaseTime = 0;
unsigned long restStart = 0;

// =====================================================
// INIT MPU
// =====================================================

void initMPU() {
  Wire.begin();
  Wire.setClock(50000);   // 50kHz: very tolerant of BT interrupts
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  delay(100);
}

// =====================================================
// READ SENSOR
// =====================================================

void readSensor(float &x, float &y, float &z, float &mag) {
  int16_t ax, ay, az;
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)6, true);
  ax = Wire.read() << 8 | Wire.read();
  ay = Wire.read() << 8 | Wire.read();
  az = Wire.read() << 8 | Wire.read();
  x   = (ax / 16384.0f) * 10.0f;
  y   = (ay / 16384.0f) * 10.0f;
  z   = (az / 16384.0f) * 10.0f;
  mag = sqrt(x*x + y*y + z*z);
}

// =====================================================
// FALL ALERT — BT push #1 (automatic)
// =====================================================

void triggerFallAlert() {

  Serial.println("================================");
  Serial.println("FALL DETECTED!");
  Serial.println("================================");

  // Push to BT immediately — only transmission that's automatic
  if (SerialBT.hasClient()) {
    SerialBT.println("FALL DETECTED");
  }

  currentActivity = "Falling";
  lastFallTime    = millis();
  fallPhase       = PHASE_IDLE;
  restStart       = 0;

  // Refill vote buffer so fall doesn't contaminate
  // next activity prediction
  initVoteBuffer("Standing");

  startBlink(5);   // non-blocking, 5 blinks
}

// =====================================================
// FALL FSM
// =====================================================

void checkFallFSM(float mag) {

  unsigned long now = millis();

  if ((now - lastFallTime) < fallCooldown) {
    fallPhase = PHASE_IDLE;
    return;
  }

  switch (fallPhase) {

    case PHASE_IDLE:
      if (mag > IMPACT_THRESH) {

        float preAvg = 0;
        int count = preBufFull ? PRE_BUF_SIZE : preBufIdx;
        if (count > 0) {
          for (int i = 0; i < count; i++) preAvg += preBuf[i];
          preAvg /= count;
        }

        if (preAvg > PRE_IMPACT_MAX) {
          Serial.println("[FSM] Rejected — jogging");
          break;
        }

        fallPhase = PHASE_IMPACT;
        phaseTime = now;
        restStart = 0;
        Serial.println("[FSM] IMPACT detected");
      }
      break;

    case PHASE_IMPACT:
      if ((now - phaseTime) > POST_REST_WINDOW_MS) {
        Serial.println("[FSM] Timeout");
        fallPhase = PHASE_IDLE;
        break;
      }

      if (mag >= POST_REST_MIN && mag <= POST_REST_MAX) {
        if (restStart == 0) restStart = now;
        if ((now - restStart) >= POST_REST_HOLD_MS) {
          Serial.println("[FSM] CONFIRMED");
          triggerFallAlert();
        }
      } else {
        restStart = 0;
      }
      break;
  }
}

// =====================================================
// ADD SAMPLE
// =====================================================

void addSample() {

  float x, y, z, mag;
  readSensor(x, y, z, mag);

  if (realSampleCount == 0) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
      sensorWindow[i][0] = x;
      sensorWindow[i][1] = y;
      sensorWindow[i][2] = z;
      sensorWindow[i][3] = mag;
    }
    realSampleCount = WINDOW_SIZE;
  } else {
    for (int i = 0; i < WINDOW_SIZE - 1; i++)
      for (int j = 0; j < CHANNELS; j++)
        sensorWindow[i][j] = sensorWindow[i + 1][j];

    sensorWindow[WINDOW_SIZE - 1][0] = x;
    sensorWindow[WINDOW_SIZE - 1][1] = y;
    sensorWindow[WINDOW_SIZE - 1][2] = z;
    sensorWindow[WINDOW_SIZE - 1][3] = mag;
  }

  preBuf[preBufIdx] = mag;
  preBufIdx = (preBufIdx + 1) % PRE_BUF_SIZE;
  if (preBufIdx == 0) preBufFull = true;

  checkFallFSM(mag);
}

// =====================================================
// INFERENCE
// =====================================================

void runInference() {

  // Build input tensor
  int k = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    for (int j = 0; j < CHANNELS; j++) {
      float val = (sensorWindow[i][j] - mean[j]) / stdv[j];
      int8_t q  = (int8_t)(val / input->params.scale + input->params.zero_point);
      if (q >  127) q =  127;
      if (q < -128) q = -128;
      input->data.int8[k++] = q;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  // ── Find best non-fall class (skip index 0) ──
  int    best       = 1;
  int    second     = 2;
  int8_t best_val   = output->data.int8[1];
  int8_t second_val = output->data.int8[2];

  for (int i = 2; i < NUM_CLASSES; i++) {
    int8_t v = output->data.int8[i];
    if (v > best_val) {
      second = best; second_val = best_val;
      best   = i;    best_val   = v;
    } else if (v > second_val) {
      second = i; second_val = v;
    }
  }

  int confidenceGap = (int)best_val - (int)second_val;

  // ── Only push to vote buffer if confident ──
  // Low confidence readings are simply ignored,
  // not fed back — this kills the Jogging lock-in
  if (confidenceGap >= 10) {
    voteBuffer[voteIdx] = labels[best];
    voteIdx = (voteIdx + 1) % VOTE_SIZE;
  }
  // (if confidenceGap < 10, vote buffer is unchanged
  //  so the majority stays what it was)

  // ── Majority vote decides the activity ──
  String voted = getMajority();

  // Don't overwrite "Falling" until display time expires
  if (currentActivity == "Falling") {
    if ((millis() - lastFallTime) > fallDisplayTime)
      currentActivity = voted;
  } else {
    currentActivity = voted;
  }

  // ── Wired serial debug ──
  float maxMag = 0, minMag = 9999;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    float m = sensorWindow[i][3];
    if (m > maxMag) maxMag = m;
    if (m < minMag) minMag = m;
  }

  Serial.print("Activity: "); Serial.print(currentActivity);
  Serial.print("  raw=");     Serial.print(labels[best]);
  Serial.print("  gap=");     Serial.print(confidenceGap);
  Serial.print("  maxMag=");  Serial.print(maxMag);
  Serial.print("  swing=");   Serial.println(maxMag - minMag);
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);

  pinMode(2, OUTPUT);
  digitalWrite(2, LOW);

  for (int i = 0; i < PRE_BUF_SIZE; i++) preBuf[i] = 0;

  initVoteBuffer("Standing");

  initMPU();

  SerialBT.begin("ESP32_FALL_DETECT");

  resolver.AddConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMean();
  resolver.AddExpandDims();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddMaxPool2D();

  static tflite::MicroInterpreter static_interpreter(
    tflModel, resolver,
    tensor_arena, tensor_arena_size,
    &micro_error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("================================");
  Serial.println("ESP32 FALL DETECTOR READY");
  Serial.println("================================");
  Serial.println("BT: send 's' to query activity");

  digitalWrite(2, HIGH); delay(300); digitalWrite(2, LOW);
}

// =====================================================
// LOOP
// =====================================================

void loop() {

  updateLED();

  if (millis() - lastSampleTime >= sampleInterval) {
    lastSampleTime = millis();
    addSample();
    runInference();
  }

  // ── Wired serial 's' command ──
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 's') {
      Serial.print(">> Activity: ");
      Serial.println(currentActivity);
    }
  }

  // ── BT: 's' query reply ONLY — no streaming ──
  if (SerialBT.available()) {
    char c = SerialBT.read();
    if (c == 's') {
      // BT push #2 — manual query
      SerialBT.print("Activity: ");
      SerialBT.println(currentActivity);
    }
  }
}
