"""
EcoSync — Feature 3: Ambulance Emergency Detection Module
Multi-modal detection (siren lights + shape + audio) + alert broadcasting
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

SIREN_DETECTION_THRESHOLD = 500        # Pixel count for red+blue combined
AMBULANCE_LOOK_BACK_SECONDS = 120      # 2 minutes upstream window
MIN_SIREN_FLASH_FRAMES = 3             # At least 3 frames of siren detected

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleTrajectory:
    """Record of a vehicle's position over time"""
    vehicle_id: str
    positions: List[Tuple[int, int]]     # [(x, y), ...]
    timestamps: List[float]               # Unix timestamps
    frame_ids: List[int]                  # Frame indices
    yolo_class: str = ""
    is_ambulance: bool = False

@dataclass
class AmbulanceAlert:
    """Emergency alert triggered when ambulance detected"""
    alert_id: str
    ambulance_position: Tuple[int, int]   # (x, y)
    vehicles_to_notify: List[str]         # Vehicle IDs upstream
    notification_count: int                # How many vehicles
    timestamp: float
    urgency: str                           # "CRITICAL" | "HIGH"
    message: str = "🚨 AMBULANCE APPROACHING"
    estimated_distance_to_signal_m: int = 200

# ═══════════════════════════════════════════════════════════════════════════════
# AMBULANCE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AmbulanceDetector:
    """
    Multi-modal ambulance detection:
    1. Siren lights (red/blue flashing in HSV space)
    2. YOLO class detection ("ambulance", "emergency_vehicle")
    3. Vehicle trajectory pattern (upstream motion in emergency)
    
    Main API:
      - update_trajectories(detections): Track all vehicles
      - detect_ambulance(frame, detections): Check for ambulance
      - get_upstream_vehicles(): Get vehicles to notify
    """
    
    def __init__(self):
        self.vehicle_trajectories: Dict[str, VehicleTrajectory] = {}
        self.ambulance_history = deque(maxlen=100)  # Last 100 detections
        self.recent_alerts = deque(maxlen=10)       # Last 10 alerts
        self.siren_flash_history = deque(maxlen=10) # Last 10 frames of siren detection
        self.detected_ambulance = False
        self.last_ambulance_time = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIREN LIGHT DETECTION (Red/Blue Flashing)
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_siren_lights(self, frame: np.ndarray) -> Tuple[int, int, float]:
        """
        Detect red and blue siren lights in frame using HSV color space.
        
        INPUT: BGR frame from OpenCV
        OUTPUT: (red_pixel_count, blue_pixel_count, combined_score)
        """
        if frame is None or frame.size == 0:
            return 0, 0, 0.0
        
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except:
            return 0, 0, 0.0
        
        # Red siren lights (0-10, 170-180 in HSV hue)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # Blue siren lights (100-130 in HSV hue)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        red_count = cv2.countNonZero(mask_red)
        blue_count = cv2.countNonZero(mask_blue)
        combined = red_count + blue_count
        
        # Score: combined pixel count normalized
        score = min(combined / max(1, SIREN_DETECTION_THRESHOLD), 1.0)
        
        return red_count, blue_count, score
    
    def detect_siren_flashing_pattern(self) -> bool:
        """
        Detect flashing pattern (siren lights appearing/disappearing).
        Uses recent siren detection history to identify flashing.
        
        OUTPUT: True if detected flashing (minimum 3 frames with siren)
        """
        if len(self.siren_flash_history) < MIN_SIREN_FLASH_FRAMES:
            return False
        
        # Check if we have at least 3 detections in last 10 frames
        recent_detections = sum(1 for s in list(self.siren_flash_history)[-10:] if s)
        return recent_detections >= MIN_SIREN_FLASH_FRAMES
    
    # ─────────────────────────────────────────────────────────────────────────
    # YOLO CLASS DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_by_yolo_class(self, detections: List[dict]) -> Tuple[bool, Optional[str]]:
        """
        Check if any detection is explicitly marked as ambulance/emergency.
        
        INPUT: List of YOLO detections [{class, confidence, bbox, ...}, ...]
        OUTPUT: (is_ambulance: bool, vehicle_id_if_ambulance: str)
        """
        for det in detections:
            yolo_class = det.get('class', '').lower()
            if any(keyword in yolo_class for keyword in 
                   ['ambulance', 'emergency', 'emergency_vehicle']):
                return True, det.get('id', 'unknown')
        
        return False, None
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAJECTORY TRACKING
    # ─────────────────────────────────────────────────────────────────────────
    
    def update_trajectories(self, detections: List[dict], frame_id: int, 
                           timestamp: float):
        """
        Update vehicle position tracking.
        Maintains history of all vehicles' positions for trajectory analysis.
        
        INPUT:
          - detections: List of current frame detections
          - frame_id: Current frame number
          - timestamp: Unix timestamp
        """
        for det in detections:
            vehicle_id = det.get('id', f"v_{frame_id}_{int(timestamp*1000)%100000}")
            center = det.get('center', (0, 0))
            yolo_class = det.get('class', '')
            
            if vehicle_id not in self.vehicle_trajectories:
                self.vehicle_trajectories[vehicle_id] = VehicleTrajectory(
                    vehicle_id=vehicle_id,
                    positions=[],
                    timestamps=[],
                    frame_ids=[],
                    yolo_class=yolo_class
                )
            
            traj = self.vehicle_trajectories[vehicle_id]
            traj.positions.append(center)
            traj.timestamps.append(timestamp)
            traj.frame_ids.append(frame_id)
            
            # Keep only last 30 positions (trajectory history)
            if len(traj.positions) > 30:
                traj.positions = traj.positions[-30:]
                traj.timestamps = traj.timestamps[-30:]
                traj.frame_ids = traj.frame_ids[-30:]
    
    def get_vehicle_velocity(self, vehicle_id: str) -> float:
        """
        Calculate vehicle velocity (pixels per second) from trajectory.
        Indicates if vehicle is moving toward signal.
        
        OUTPUT: Pixels per second (positive = moving right/down)
        """
        if vehicle_id not in self.vehicle_trajectories:
            return 0.0
        
        traj = self.vehicle_trajectories[vehicle_id]
        if len(traj.positions) < 2:
            return 0.0
        
        # Use last 5 positions for smoothing
        positions = traj.positions[-5:]
        timestamps = traj.timestamps[-5:]
        
        if len(positions) < 2:
            return 0.0
        
        # Average y-velocity (assuming signal is at bottom of frame)
        y_deltas = [positions[i+1][1] - positions[i][1] for i in range(len(positions)-1)]
        time_deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        velocities = [y_deltas[i] / max(0.001, time_deltas[i]) 
                     for i in range(len(y_deltas))]
        
        return np.mean(velocities) if velocities else 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_ambulance(self, frame: np.ndarray, detections: List[dict],
                        frame_id: int = 0) -> bool:
        """
        Multi-modal ambulance detection combining three methods:
        
        INPUT:
          - frame: BGR image frame
          - detections: List of YOLO detections
          - frame_id: Current frame index
        
        OUTPUT: True if ambulance detected
        """
        ambulance_detected = False
        detection_method = ""
        
        # METHOD 1: Siren lights detection
        red_px, blue_px, siren_score = self.detect_siren_lights(frame)
        siren_detected = (red_px + blue_px) > SIREN_DETECTION_THRESHOLD
        self.siren_flash_history.append(siren_detected)
        
        if siren_detected:
            ambulance_detected = True
            detection_method = f"Siren (R:{red_px} B:{blue_px})"
        
        # METHOD 2: YOLO class detection
        yolo_ambulance, ambulance_id = self.detect_by_yolo_class(detections)
        if yolo_ambulance:
            ambulance_detected = True
            detection_method = f"YOLO Class ({ambulance_id})"
        
        # METHOD 3: Trajectory analysis (optional: upstreaming fast)
        # Can add pattern matching for emergency vehicle behavior
        
        if ambulance_detected:
            self.detected_ambulance = True
            self.last_ambulance_time = time.time()
            self.ambulance_history.append({
                'frame_id': frame_id,
                'timestamp': time.time(),
                'method': detection_method,
                'siren_score': siren_score
            })
        
        return ambulance_detected
    
    # ─────────────────────────────────────────────────────────────────────────
    # UPSTREAM VEHICLE IDENTIFICATION & ALERTING
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_upstream_vehicles(self, ambulance_position: Tuple[int, int],
                             current_timestamp: float) -> List[str]:
        """
        Find all vehicles upstream (before/ahead of ambulance).
        Uses 2-minute lookback window to identify vehicles to notify.
        
        INPUT:
          - ambulance_position: (x, y) center of ambulance
          - current_timestamp: Unix timestamp
        
        OUTPUT: List of vehicle IDs to notify
        """
        cutoff_time = current_timestamp - AMBULANCE_LOOK_BACK_SECONDS
        ambulance_y = ambulance_position[1]
        
        upstream_vehicles = []
        
        for vehicle_id, traj in self.vehicle_trajectories.items():
            if not traj.positions or not traj.timestamps:
                continue
            
            # Filter by time window
            recent_positions = [
                (pos, ts) for pos, ts in zip(traj.positions, traj.timestamps)
                if ts > cutoff_time
            ]
            
            if not recent_positions:
                continue
            
            # Check if vehicle is upstream (higher y = closer to bottom signal)
            # Assume ambulance coming from behind (higher y value)
            avg_y = np.mean([pos[1] for pos, _ in recent_positions])
            
            if avg_y > ambulance_y:  # Upstream vehicles are ahead (lower y)
                upstream_vehicles.append(vehicle_id)
        
        return upstream_vehicles
    
    def generate_alert(self, ambulance_position: Tuple[int, int],
                      upstream_vehicles: List[str],
                      frame_id: int) -> AmbulanceAlert:
        """
        Generate alert payload for broadcasting.
        
        OUTPUT: Alert object ready for WebSocket distribution
        """
        alert_id = f"AMB_{int(time.time())}_{frame_id}"
        
        alert = AmbulanceAlert(
            alert_id=alert_id,
            ambulance_position=ambulance_position,
            vehicles_to_notify=upstream_vehicles,
            notification_count=len(upstream_vehicles),
            timestamp=time.time(),
            urgency="CRITICAL",
            message="🚨 AMBULANCE APPROACHING - CLEAR THE PATH",
            estimated_distance_to_signal_m=200
        )
        
        self.recent_alerts.append(alert)
        return alert
    
    # ─────────────────────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_status(self) -> dict:
        """Get current ambulance detection status"""
        return {
            'ambulance_detected': self.detected_ambulance,
            'last_detection_time': self.last_ambulance_time,
            'recent_detections_count': len(self.ambulance_history),
            'tracked_vehicles': len(self.vehicle_trajectories),
            'recent_alerts': [
                {
                    'id': alert.alert_id,
                    'vehicles_notified': alert.notification_count,
                    'timestamp': alert.timestamp
                }
                for alert in list(self.recent_alerts)[-5:]
            ]
        }
    
    def reset(self):
        """Reset detector state"""
        self.detected_ambulance = False
        self.vehicle_trajectories.clear()
        self.siren_flash_history.clear()

