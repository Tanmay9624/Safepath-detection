"""
SafePath AI: Priority Audio Engine Test Suite
============================================
Comprehensive test suite verifying:
  - Case 1 & 2: Priority Preemption & Immediate Interruption (Priority 1 cuts off Priority 3).
  - Case 3: Debouncing & Spam Filtering (Suppresses repeated alerts within 3s cooldown).
  - Case 4: Dynamic Spatial Override (Proximity breach <=1.5m or delta >=2.0m overrides cooldown).
  - Case 5: System-Level Fatal Error Injection (Priority 0 system alert).
"""

import os
import sys
import time

# Ensure parent directory (full_test) is in sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from audio_engine import PriorityAudioEngine

def run_comprehensive_tests():
    print("==================================================================")
    print(" SafePath AI: Running Priority Audio Engine Test Suite")
    print("==================================================================")
    
    audio = PriorityAudioEngine(default_cooldown=3.0)
    
    # ---------------------------------------------------------
    # CASE 1 & 2: Preemption & Hardware Interruption
    # Expected: The first sentence starts but is cut off 
    # at 1.5 seconds by an earcon beep and the urgent bicycle warning.
    # ---------------------------------------------------------
    print("\n[TEST 1 & 2] Testing Priority Preemption & Immediate Alert...")
    audio.add_alert({
        "object_id": "nav_01",
        "priority": 3,
        "message": "Walkable path is clear. Continue straight ahead for the next ten meters.",
        "hazard_type": "navigation",
        "distance_m": 0.0
    })
    time.sleep(1.5) 
    
    audio.add_alert({
        "object_id": "bike_01",
        "priority": 1,
        "message": "Stop, bicycle crossing",
        "hazard_type": "vehicle",
        "distance_m": 1.2
    })
    time.sleep(4.0) 
    
    # ---------------------------------------------------------
    # CASE 3: Debouncing & Spam Filtering
    # Expected: The system speaks "Wall on your right" exactly once,
    # completely suppressing the next 4 rapid-fire payloads.
    # ---------------------------------------------------------
    print("\n[TEST 3] Testing Cooldown Debouncing & Spam Filtering...")
    for i in range(5):
        audio.add_alert({
            "object_id": "wall_01",
            "priority": 2,
            "message": "Wall on your right",
            "hazard_type": "obstacle",
            "distance_m": 4.0
        })
        time.sleep(0.2) 
    
    time.sleep(1.5) 
    
    # ---------------------------------------------------------
    # CASE 4: Dynamic Spatial Override
    # Expected: Even though the 3-second cooldown hasn't finished, 
    # the distance drops from 4.0m to 1.0m (a delta >= 2.0m). 
    # This triggers the critical zone override and speaks immediately.
    # ---------------------------------------------------------
    print("\n[TEST 4] Testing Dynamic Spatial Override (Delta >= 2.0m & Near Zone <= 1.5m)...")
    audio.add_alert({
        "object_id": "wall_01",
        "priority": 1,
        "message": "Brace, wall directly ahead",
        "hazard_type": "obstacle",
        "distance_m": 1.0
    })
    time.sleep(3.0)

    # ---------------------------------------------------------
    # CASE 5: System-Level Fatal Error Injection
    # Expected: A Priority 0 alert announces a hardware/system failure.
    # ---------------------------------------------------------
    print("\n[TEST 5] Testing System-Level Fatal Error Injection (Priority 0)...")
    audio.add_system_error("Critical error. Camera feed disconnected. Please stop.")
    
    time.sleep(5.0)
    print("\n[COMPLETE] All audio engine test cases executed successfully.")

if __name__ == "__main__":
    run_comprehensive_tests()

