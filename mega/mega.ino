//////////////////////////////
// Mega Sensor System
//////////////////////////////

// Pins
#define PIR_PIN 2         // PIR motion sensor
#define IR_PIN 3          // IR obstacle sensor
#define TRIG_PIN 4        // Ultrasonic TRIG
#define ECHO_PIN 5        // Ultrasonic ECHO

//////////////////////////////
// Setup
//////////////////////////////
void setup() {
  Serial.begin(9600);     // For Serial Monitor
  Serial1.begin(9600);    // For ESP32

  pinMode(PIR_PIN, INPUT);
  pinMode(IR_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  Serial.println("=== Mega Sensor System Initialized ===");
}

//////////////////////////////
// Ultrasonic Distance Function
//////////////////////////////
long readUltrasonicCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  if (duration == 0) return -1; // No echo detected
  long distanceCM = duration * 0.034 / 2;
  return distanceCM;
}

//////////////////////////////
// Main Loop
//////////////////////////////
void loop() {
  // Read sensors
  int pirState = digitalRead(PIR_PIN);       // 0 = No Motion, 1 = Motion
  int irState = digitalRead(IR_PIN);         // 0 = Obstacle, 1 = Clear
  long distance = readUltrasonicCM();        // Distance in cm

  // Print to Serial Monitor
  Serial.println("=== Sensor Data ===");
  Serial.print("PIR Sensor: "); Serial.println(pirState ? "Motion Detected" : "No Motion");
  Serial.print("IR Sensor: "); Serial.println(irState ? "No Obstacle" : "Obstacle Detected");
  if(distance != -1) {
  Serial.print("Ultrasonic Distance: "); 
  Serial.println(distance);
} else {
  Serial.println("Ultrasonic Distance: Out of Range");
}


  // Send JSON to ESP32
  char jsonBuffer[100];
  snprintf(jsonBuffer, sizeof(jsonBuffer),
           "{\"pir\":%d,\"ir\":%d,\"ultrasonic\":%ld}",
           pirState, irState, distance);
  Serial1.println(jsonBuffer);
  Serial.print("JSON Sent: "); Serial.println(jsonBuffer);

  Serial.println("----------------------------");

  delay(1000); // 1-second delay
}
