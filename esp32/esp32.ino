#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <WiFiManager.h>  // WiFiManager handles auto WiFi connection

// LED indicator
#define LED_PIN 2  // Onboard LED

// Serial2 pins (for Mega → ESP32)
#define RX2_PIN 16
#define TX2_PIN 17

// FastAPI cloud endpoint
const char* SERVER_URL = "https://esp32-fastapi-server-uh47.onrender.com/data";

WiFiManager wm;
StaticJsonDocument<256> doc;
bool wifiConnected = false;

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600, SERIAL_8N1, RX2_PIN, TX2_PIN);
  pinMode(LED_PIN, OUTPUT);

  Serial.println("\nBooting ESP32 Sensor Bridge...");

  // Try to connect using saved Wi-Fi credentials
  if (!wm.autoConnect("ESP32_Setup")) {
    Serial.println("WiFi AutoConnect failed, please configure manually!");
  } else {
    wifiConnected = true;
    Serial.print("WiFi connected! IP: ");
    Serial.println(WiFi.localIP());
  }
}

void loop() {
  if (Serial2.available()) {
    char buffer[150];
    size_t len = Serial2.readBytesUntil('\n', buffer, sizeof(buffer) - 1);
    buffer[len] = '\0';

    if (len < 3) return;

    DeserializationError error = deserializeJson(doc, buffer);
    if (error) {
      Serial.print("❌ JSON Parse Error: ");
      Serial.println(error.c_str());
      return;
    }

    // Extract values
    int pir = doc["pir"];
    int ir = doc["ir"];
    long ultrasonic = doc["ultrasonic"];

    // Print to Serial Monitor
    Serial.println("=== Mega Sensor Dashboard ===");
    Serial.printf("PIR: %s\n", pir ? "Motion Detected" : "No Motion");
    Serial.printf("IR : %s\n", ir ? "No Obstacle" : "Obstacle Detected");
    Serial.printf("Ultrasonic: %ld cm\n", ultrasonic);
    Serial.println("-----------------------------");

    // Blink LED
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);

    // Send to cloud if connected
    if (wifiConnected && WiFi.status() == WL_CONNECTED) {
      sendToCloud(buffer);
    } else {
      Serial.println(" WiFi not connected. Skipping cloud update.");
    }

    doc.clear();
  }

  delay(100);
}

void sendToCloud(const char* jsonData) {
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  int httpResponseCode = http.POST((uint8_t*)jsonData, strlen(jsonData));

  if (httpResponseCode > 0) {
    Serial.printf("Sent to cloud! Response: %d\n", httpResponseCode);
  } else {
    Serial.printf("Failed to send! Error: %s\n", http.errorToString(httpResponseCode).c_str());
  }

  http.end();
}
