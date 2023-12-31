//-------------------------------------------------------
//     LSB  P0  P1  P2   P3   P4   P5,  P6,  P7  MSB
//-------------------------------------------------------
// PORTA = {22, 23, 24,  25,  26,  27,  28,  29};        0
// PORTF = {A0, A1, A2,  A3,  A4,  A5,  A6,  A7};        1
// PORTK = {A8, A9, A10, A11, A12, A13, A14, A15};       2
// PORTC = {37, 36, 35,  34,  33,  32,  31,  30};        3
// PORTL = {49, 48, 47,  46,  45,  44,  43,  42};        4
// PORTB = {53, 52, 51,  50,  10,  11,  12,  13};        5
// PORTG = {41, 40, 39,  xx,  xx,  4,   xx,  xx};        6
// PORTD = {21, 20, 19,  18,  xx,  xx,  xx,  38};        7
// PORTE = {0,  1,  xx,  5,   2,   3,   xx,  xx};        8
// PORTH = {17, 16, xx,  6,   7,   8,   9,   xx};        9
// PORTJ = {15, 14, xx,  xx,  xx,  xx,  xx,  xx};        10
//-------------------------------------------------------

// WR_REG: PORTA; PORTF; PORTK; PORTG;
// CLK: 39
// MOSI: 40
// 25CS: 41

// READ_REG: PORTC; PORTL; PORTB; PORTJ(0)

#define NOP __asm__ __volatile__ ("nop\n\t")

int test_buffer = 0;

void setup() {
  // put your setup code here, to run once:
  DDRA = 0b11111111;
  DDRF = 0b11111111;
  DDRK = 0b11111111;
  DDRG = 0b00000111;

  DDRC = 0b00000000;
  DDRL = 0b00000000;
  DDRB = 0b00000000;
  DDRJ = 0b00000000;
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  // read_individual(0b01001001, 1, test_buffer, 0, 25);
  test_buffer = 0;
  read_individual(0xD0, 1, test_buffer, 1, 25);
  Serial.println(test_buffer, HEX);
  delay(2000);
}

void read_individual(uint8_t wr_buffer, size_t wr_buf_len, int &rd_buffer, size_t rd_buf_len, uint16_t wr_reg_index) {
  switch (wr_reg_index / 8) {
    case 0: PORTA = PORTA & (0b11111111 ^ (1 << ((wr_reg_index - 1) % 8))); break;
    case 1: PORTF = PORTF & (0b11111111 ^ (1 << ((wr_reg_index - 1) % 8))); break;
    case 2: PORTK = PORTK & (0b11111111 ^ (1 << ((wr_reg_index - 1) % 8))); break;
    case 3: PORTG = PORTG & (0b11111111 ^ (1 << ((wr_reg_index - 1) % 8))); break;
  }
  for (uint8_t itt = 0; itt < wr_buf_len * 8; itt++) {
    if ((wr_buffer & (1 << wr_buf_len * 8 - 1 - itt)) > 0){
      PORTG = PORTG | 0b00000010;
    }else{
      PORTG = PORTG & 0b11111101;
    }
    // delay(250);
    NOP;
    PORTG = PORTG | 0b00000100;
    NOP;
    // delay(250);
    PORTG = PORTG & 0b11111001;
    // delay(500);
    // wr_buffer <<= 1;
  }
  for (uint8_t itt = 0; itt < rd_buf_len * 8; itt++) {
    PORTG = PORTG | 0b00000100;
    NOP;
    // delay(500);
    switch (wr_reg_index / 8) {
      case 0: rd_buffer |= (PINC & (1 << ((wr_reg_index - 1) % 8))) << (rd_buf_len * 8 - 1 - itt); break;
      case 1: rd_buffer |= (PINL & (1 << ((wr_reg_index - 1) % 8))) << (rd_buf_len * 8 - 1 - itt); break;
      case 2: rd_buffer |= (PINB & (1 << ((wr_reg_index - 1) % 8))) << (rd_buf_len * 8 - 1 - itt); break;
      case 3: rd_buffer |= (PINJ & 0b00000001) << (rd_buf_len * 8 - 1 - itt); break;
    }
    PORTG = PORTG & 0b11111001;
    NOP;
    // delay(500);
    // rd_buffer = rd_buffer << 1;
    // Serial.println(rd_buffer, BIN);
  }
  PORTG = PORTG & 0b11111000;
  switch (wr_reg_index / 8) {
    case 0: PORTA = PORTA | (1 << ((wr_reg_index - 1) % 8)); break;
    case 1: PORTF = PORTF | (1 << ((wr_reg_index - 1) % 8)); break;
    case 2: PORTK = PORTK | (1 << ((wr_reg_index - 1) % 8)); break;
    case 3: PORTG = PORTG | (1 << ((wr_reg_index - 1) % 8)); break;
  }
}
