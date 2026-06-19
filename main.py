import cv2
import torch
import traceback
import time
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

# Vision imports
from vision.inference import load_model, infer_count_and_density
from vision.camera_capture import capture_frame_from_webcam

# Sensor imports
from sensor.weight_sensor import WeightSensor
from sensor.calibration import weight_to_count
from sensor.filter import smooth_weight

# Fusion imports
from fusion.decision import decide
from fusion.tolerance import get_tolerance

# --- CONFIGURATION CONSTANTS ---
MODEL_WEIGHTS_PATH = "weights/best_model.pth"
CAMERA_INDEX = 0
UNIT_WEIGHT = 3.0
SIMULATE_SENSOR = True
BUFFER_SIZE = 5
SERIAL_PORT = 'COM3'
BAUDRATE = 9600

def log(level: str, message: str) -> None:
    """Simple structured logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {message}")

class InspectionSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sensor = None
        self.weight_buffer = []

    def initialize(self):
        """Setup model and sensor gracefully."""
        log("INFO", "Starting system initialization...")
        
        # 1. Initialize Vision Model
        try:
            self.model = load_model(MODEL_WEIGHTS_PATH, self.device)
            log("INFO", "Vision model loaded successfully.")
        except Exception as e:
            log("WARNING", f"Vision model loading failed: {e}. Running without vision model.")
            self.model = None

        # 2. Initialize Weight Sensor
        global SIMULATE_SENSOR
        try:
            self.sensor = WeightSensor(port=SERIAL_PORT, baudrate=BAUDRATE)
            if not SIMULATE_SENSOR:
                self.sensor.connect_serial()
                log("INFO", "Connected to hardware weight sensor.")
        except Exception as e:
            log("ERROR", f"Hardware sensor initialization failed: {e}")
            log("WARNING", "Falling back to sensor simulation.")
            SIMULATE_SENSOR = True
            
        log("INFO", "System initialization complete.")

    def capture_phase(self) -> Optional[np.ndarray]:
        """Captures a frame from the camera."""
        try:
            return capture_frame_from_webcam(CAMERA_INDEX)
        except Exception as e:
            log("ERROR", f"Camera capture error: {e}")
            return None

    def vision_phase(self, frame: np.ndarray) -> Optional[float]:
        """Runs the vision inference to get the predicted count."""
        if self.model is None:
            return None
        try:
            model_count, _ = infer_count_and_density(frame, self.model, self.device)
            return model_count
        except Exception as e:
            log("ERROR", f"Vision inference error: {e}")
            return None

    def sensor_phase(self) -> Tuple[Optional[float], Optional[float]]:
        """Reads weight, applies smoothing, and converts to estimated count."""
        try:
            current_weight = self.sensor.read_weight(simulate=SIMULATE_SENSOR)
            
            if current_weight is None:
                log("WARNING", "No weight reading received.")
                return None, None
                
            self.weight_buffer.append(current_weight)
            if len(self.weight_buffer) > BUFFER_SIZE:
                self.weight_buffer.pop(0)
                
            smoothed_weight = smooth_weight(self.weight_buffer, window_size=BUFFER_SIZE, use_ema=True)
            
            if smoothed_weight is None:
                return None, None
                
            weight_count = weight_to_count(smoothed_weight, unit_weight=UNIT_WEIGHT, round_result=True)
            return smoothed_weight, weight_count
        except Exception as e:
            log("ERROR", f"Sensor processing error: {e}")
            return None, None

    def decision_phase(self, weight_count: Optional[float], model_count: Optional[float]) -> Tuple[str, Optional[float]]:
        """Makes the fusion decision based on sensor and vision outputs."""
        try:
            tolerance = get_tolerance()
            return decide(weight_count, model_count, tolerance)
        except Exception as e:
            log("ERROR", f"Decision logic error: {e}")
            return "UNKNOWN", None

    def display_results(self, frame: np.ndarray, model_count: Optional[float], 
                        smoothed_weight: Optional[float], weight_count: Optional[float], 
                        status: str, diff: Optional[float]) -> bool:
        """Renders the UI and checks for exit commands."""
        try:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            m_count_str = f"{model_count:.1f}" if model_count is not None else "N/A"
            cv2.putText(display_frame, f"Model Count: {m_count_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            w_count_str = f"{weight_count:.1f}" if weight_count is not None else "N/A"
            s_weight_str = f"{smoothed_weight:.1f}g" if smoothed_weight is not None else "N/A"
            cv2.putText(display_frame, f"Weight Count: {w_count_str} (W: {s_weight_str})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            status_color = (0, 255, 0) if status == "OK" else ((0, 0, 255) if status == "NG" else (128, 128, 128))
            diff_str = f"{diff:.1f}" if diff is not None else "N/A"
            cv2.putText(display_frame, f"STATUS: {status} (Diff: {diff_str})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            cv2.imshow("Industrial Inspection System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
            return True
        except Exception as e:
            log("ERROR", f"Display rendering error: {e}")
            return True # Continue loop even if display fails

    def run(self):
        """Main robust execution loop."""
        self.initialize()
        log("INFO", "Starting system loop. Press 'q' in OpenCV window or Ctrl+C to exit.")
        
        while True:
            try:
                # 1. Capture Phase
                frame = self.capture_phase()
                if frame is None:
                    time.sleep(1)
                    continue
                    
                # 2. Vision Phase
                model_count = self.vision_phase(frame)
                
                # 3. Sensor Phase
                smoothed_weight, weight_count = self.sensor_phase()
                
                # 4. Decision Phase
                status, diff = self.decision_phase(weight_count, model_count)
                
                # 5. Display Review Phase
                continue_loop = self.display_results(frame, model_count, smoothed_weight, weight_count, status, diff)
                
                if not continue_loop:
                    log("INFO", "Exit sequence initiated by user.")
                    break
                    
                time.sleep(0.1) # Throttle loop rate
                
            except KeyboardInterrupt:
                log("INFO", "Keyboard interrupt received. Exiting.")
                break
            except Exception as e:
                log("ERROR", f"Critical failure in main loop: {e}\n{traceback.format_exc()}")
                time.sleep(1) # Prevent hot looping on failure

        cv2.destroyAllWindows()
        log("INFO", "System shut down gracefully.")

def main():
    system = InspectionSystem()
    system.run()

if __name__ == "__main__":
    main()
