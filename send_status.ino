/*
  Upload fixed plant-status JSON to an S3 object
  — ESP32 version that mirrors the provided Python script
  -------------------------------------------------------
  Libraries:
    • WiFi            (incluida con ESP32 core)
    • WiFiClientSecure(incluida con ESP32 core)
    • HTTPClient      (incluida con ESP32 core)
    • ArduinoJson ≥6  (Instálala desde el Library Manager)
*/

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

constexpr char WIFI_SSID[] = "UM_WiFi";
constexpr char WIFI_PASS[] = "umontevideo";

constexpr char BUCKET[]     = "303702502406-ia-plant";
constexpr char REGION[]     = "us-east-1";
constexpr char OBJECT_KEY[] = "last_value.json";

constexpr char SOIL[]   = "1444";
constexpr char LIGHT[]  = "785";
constexpr char TEMP_C[] = "23.0";
constexpr char RESULT[] = "Marchitaaaab";

// ─── Intervalo de subida ──────────────────────────────────────────────────────
constexpr unsigned long UPLOAD_INTERVAL_MS = 30UL * 60UL * 1000UL; // 30 min

// ─── Prototipos ───────────────────────────────────────────────────────────────
void connectWiFi();
String buildJson();
void uploadToS3(const String& url, const String& payload);

// ─── Variables de estado ──────────────────────────────────────────────────────
unsigned long lastUpload = 0; // marca de tiempo de la última subida

void setup() {
  Serial.begin(115200);
  delay(1200);

  connectWiFi();

  String url = String("https://") + BUCKET + ".s3." + REGION +
               ".amazonaws.com/" + OBJECT_KEY;

  uploadToS3(url, buildJson());
  lastUpload = millis();
}

void loop() {
  // Comprueba si han pasado 30 minutos
  if (millis() - lastUpload >= UPLOAD_INTERVAL_MS) {
    String url = String("https://") + BUCKET + ".s3." + REGION +
                 ".amazonaws.com/" + OBJECT_KEY;
    uploadToS3(url, buildJson());
    lastUpload = millis();

}

void connectWiFi() {
  Serial.printf("Conectando a %s…\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint8_t retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print('.');
    ++retries;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nConectado, IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nERROR: no se pudo conectar a Wi-Fi");
    ESP.restart();
  }
}

String buildJson() {
  StaticJsonDocument<128> doc;
  doc["soil"]   = SOIL;
  doc["light"]  = LIGHT;
  doc["temp_c"] = TEMP_C;
  doc["result"] = RESULT;

  String out;
  serializeJson(doc, out);
  Serial.printf("JSON generado: %s\n", out.c_str());
  return out;
}

void uploadToS3(const String& url, const String& payload) {
  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  Serial.printf("Subiendo a %s…\n", url.c_str());
  if (http.begin(client, url)) {
    http.addHeader("Content-Type", "application/json");
    int code = http.PUT((uint8_t*)payload.c_str(), payload.length());

    Serial.printf("HTTP code: %d\n", code);
    if (code > 0) {
      Serial.println(http.getString());
    }
    http.end();
  } else {
    Serial.println("ERROR: http.begin() falló");
  }
}

void connectWiFi();
String buildJson();
void uploadToS3(const String& url, const String& payload);

