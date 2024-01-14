#define STEP_PIN 9
#define DIR_PIN 8
#define SPEED_MS 1000


// int a = 0;
// int v = MAX_SPEED_MS;
// String buffer = "";
void setup() {
  // put your setup code here, to run once:
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(DIR_PIN, HIGH);
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  // if (Serial.available()){
  //   char b = Serial.read();
  //   if (b == '\n'){
  //   }else{
  //     buffer += b;
  //   }
  // }
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(SPEED_MS);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(SPEED_MS);
}
