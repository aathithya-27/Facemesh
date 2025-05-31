void setup() {
    Serial.begin(9600);  // Set baud rate to 9600
    pinMode(LED_BUILTIN, OUTPUT);  // Using built-in LED for indication
}

void loop() {
    if (Serial.available() > 0) {  // Check if data is available
        char received = Serial.read();  // Read received byte

        if (received == '1') {  // If received '1', turn on LED
            digitalWrite(LED_BUILTIN, HIGH);
            delay(1000);
            digitalWrite(LED_BUILTIN, LOW);
        }
    }
}
