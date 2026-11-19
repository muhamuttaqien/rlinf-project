// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <exception>

#include "rlinf_robotiq_observation/observation.hpp"

// Metadata and cooperative lock only: never construct a serial connection.
int main() {
  try {
    rlinf_robotiq_observation::PortClaim claim(
        "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0");
    claim.acquire();
    claim.release();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Complete ownership inspection required; found: %s\n",
                 error.what());
    return 1;
  }
}
