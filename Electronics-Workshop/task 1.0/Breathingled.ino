int i;
float delay_time;

void setup()
{
  i = 0;
  pinMode(11, OUTPUT);
}

void loop()
{
  delay_time = 10 ;
  
  for (i = 0; i <= 255; i += 5) {
    analogWrite(11, i);
    delay(delay_time);
  }

  for (i = 255; i >= 0; i -= 5) {
    analogWrite(11, i);
    delay(delay_time);
  }
    
}