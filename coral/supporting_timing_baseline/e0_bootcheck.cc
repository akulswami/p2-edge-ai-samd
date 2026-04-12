#include <cstdio>
#include "libs/base/check.h"
#include "libs/base/init.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"

extern "C" void app_main(void) {
  coral::Init();
  printf("E0 bootcheck start\r\n");

  bool on = false;
  while (true) {
    on = !on;
    coral::LedSet(coral::Led::kUser, on);
    printf("alive tick\r\n");
    coral::SleepMillis(1000);
  }
}
