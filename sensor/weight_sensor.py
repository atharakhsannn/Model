import random
import serial
import time
from typing import Optional

class WeightSensor:
    """Weight sensor interface for simulation and serial modes."""
    
    def __init__(self, port: str = 'COM3', baudrate: int = 9600):
        """Initializes the sensor config."""
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None

    def connect_serial(self) -> None:
        """Establishes a serial connection to the Arduino."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow time for connection
        except serial.SerialException as e:
            raise RuntimeError(f"Could not connect to serial port {self.port}: {e}")

    def read_weight(self, simulate: bool = True, retries: int = 3) -> Optional[float]:
        """
        Reads weight from sensor with a retry mechanism.
        If simulate is True, returns a simulated weight.
        If simulate is False, reads from COM port.
        Returns None if data cannot be read after retries.
        """
        if simulate:
            return round(random.uniform(290.0, 310.0), 2)
        
        for attempt in range(retries):
            if self.serial_conn is None or not self.serial_conn.is_open:
                try:
                    self.connect_serial()
                except RuntimeError:
                    time.sleep(1)
                    continue
                
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line.startswith("WEIGHT:"):
                        weight_str = line.split(":")[1]
                        return float(weight_str)
            except Exception:
                pass
            
            time.sleep(0.5)
            
        return None
