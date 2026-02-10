void setup() {
  pinMode(7,INPUT);
  Serial.begin(9600);
  pinMode(13,OUTPUT);
}
void loop(){
    int sensorValueDigital = digitalRead(7);
    Serial.println(sensorValueDigital);
    if (sensorValueDigital == 1 ){
      digitalWrite(13,HIGH);
      delay(2000);
      digitalWrite(13,LOW);
    } 
    delay(500);
}