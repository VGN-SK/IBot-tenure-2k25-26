bool state = false;
bool lastState = true;

void setup() {
    pinMode(2, INPUT_PULLUP);
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    bool buttonState = digitalRead(2);

    if (lastState == true && buttonState == false) {
        state = !state;
        digitalWrite(LED_BUILTIN, state);
        delay(20); 
    }

    lastState = buttonState;
}
