#!/usr/bin/env python3
"""
Send confirmed BIN to ESP32 → relay/LED lights.

Protocol (UART, newline-terminated ASCII):
  BIN <n>\\n   n in 1..4  → that relay ON, others OFF
  BIN 0\\n               → all OFF
  PING\\n                → ESP32 replies PONG

ESP32 replies: OK\\n or ERR\\n
"""

from __future__ import annotations

import json
import time
from pathlib import Path


DEFAULT_BIN_TO_RELAY = {
    "Before-DW": 1,
    "Before-UW": 2,
    "After-DW": 3,
    "After-UW": 4,
}


class BinLightController:
    def __init__(
        self,
        port: str | None,
        baud: int = 115200,
        bin_map: dict[str, int] | None = None,
        enabled: bool = True,
    ):
        self.port = port
        self.baud = baud
        self.bin_map = {str(k).strip(): int(v) for k, v in (bin_map or DEFAULT_BIN_TO_RELAY).items()}
        self.enabled = bool(enabled and port)
        self._ser = None
        self._last_relay: int | None = None

    @classmethod
    def from_map_file(cls, port: str | None, map_path: Path | None, baud: int = 115200) -> "BinLightController":
        bin_map = dict(DEFAULT_BIN_TO_RELAY)
        if map_path and map_path.exists():
            data = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                bin_map.update({str(k): int(v) for k, v in data.items()})
        return cls(port=port, baud=baud, bin_map=bin_map, enabled=bool(port))

    def open(self) -> None:
        if not self.enabled:
            print("ESP32 bin lights: disabled (no --esp32-port)")
            return
        try:
            import serial  # pyserial
        except ImportError as e:
            raise SystemExit(
                "pyserial required for ESP32 lights: pip install pyserial"
            ) from e
        self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
        time.sleep(0.4)  # ESP32 USB-serial reset settle
        self._ser.reset_input_buffer()
        print(f"ESP32 bin lights: {self.port} @ {self.baud}  map={self.bin_map}")
        self.all_off()

    def close(self) -> None:
        if self._ser is not None:
            try:
                self.all_off()
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _send(self, line: str) -> bool:
        if not self.enabled or self._ser is None:
            return False
        msg = (line.strip() + "\n").encode("ascii", errors="ignore")
        self._ser.write(msg)
        self._ser.flush()
        resp = self._ser.readline().decode("ascii", errors="ignore").strip()
        ok = resp.upper().startswith("OK") or resp.upper() == "PONG"
        if not ok:
            print(f"  [esp32] cmd={line!r} resp={resp!r}")
        return ok

    def set_relay(self, relay: int) -> None:
        """relay 0 = all off; 1..4 = that channel only."""
        relay = int(relay)
        if relay < 0 or relay > 4:
            print(f"  [esp32] invalid relay {relay}")
            return
        if self._last_relay == relay:
            return
        if self._send(f"BIN {relay}"):
            self._last_relay = relay
            print(f"  *** LIGHTS: relay {relay} ***" if relay else "  *** LIGHTS: all OFF ***")

    def all_off(self) -> None:
        self._last_relay = None
        self._send("BIN 0")
        self._last_relay = 0

    def set_bin(self, bin_label: str | None) -> None:
        if not bin_label:
            return
        key = str(bin_label).strip()
        relay = self.bin_map.get(key)
        if relay is None:
            print(f"  [esp32] no relay mapped for BIN={key!r} (map={self.bin_map})")
            return
        self.set_relay(relay)
