//
//  RegManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam {

  void RegistrationRunnable::run() {
    regInfo->attempt_absolute_reg(true);
    parent->regCount--;
  }

}
