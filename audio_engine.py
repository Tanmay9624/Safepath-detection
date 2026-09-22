import os
import sys
import threading
import queue
import time

# Optional Windows-specific audio libraries
HAVE_WINSOUND = False
HAVE_SAPI = False

if sys.platform == "win32":
    try:
        import winsound
        HAVE_WINSOUND = True
    except ImportError:
        pass
    try:
        import win32com.client
        import pythoncom
        HAVE_SAPI = True
    except ImportError:
        pass

class PriorityAudioEngine:
    """
    Priority-driven, non-blocking asynchronous TTS Audio Engine.
    Strictly adheres to teammate schema, priority preemption, and dynamic spatial debouncing:
      - Priority 0: Fatal hardware / system error (preemptive, urgent rate, 1000Hz tone)
      - Priority 1: Immediate collision hazard on path (preemptive, 1000Hz earcon beep, rate=1)
      - Priority 2: Directional steering command (VEER LEFT / VEER RIGHT, rate=-2)
      - Priority 3: Informational path status (Path clear / proceed, rate=-2, debounced)
    """
    def __init__(self, default_cooldown=3.0):
        self.alert_queue = queue.PriorityQueue()
        self.hazard_state = {}
        self.default_cooldown = default_cooldown
        self.last_spoken_message = "Ready"
        self.last_spoken_priority = 3
        
        self.worker_thread = threading.Thread(target=self._audio_worker, name="PriorityAudioWorker", daemon=True)
        self.worker_thread.start()

    def add_alert(self, payload):
        """
        Submits an alert payload to the prioritized speech queue.
        Payload Schema:
          {
             "object_id": str,
             "priority": int (0=fatal, 1=urgent, 2=directional, 3=info),
             "message": str,
             "hazard_type": str,
             "distance_m": float
          }
        """
        obj_id = payload.get("object_id", payload.get("hazard_type", "hazard"))
        current_time = time.time()
        new_dist = float(payload.get("distance_m", 0.0))
        priority = int(payload.get("priority", 3))

        # Debouncing & Dynamic Spatial Override
        if obj_id in self.hazard_state and priority > 0:
            last_time = self.hazard_state[obj_id]["time"]
            last_dist = self.hazard_state[obj_id]["distance"]
            
            # Dynamic spatial override:
            # 1. Critical proximity breach: distance dropped to <= 1.5m
            # 2. Sudden rapid approach: distance dropped by >= 2.0m
            spatial_override = ((new_dist <= 1.5 and last_dist > 1.5) or (last_dist - new_dist >= 2.0))
            
            # Use per-payload custom cooldown if specified (e.g. clear path heartbeat), else default
            effective_cooldown = float(payload.get("cooldown", self.default_cooldown))
            if not spatial_override and (current_time - last_time < effective_cooldown):
                # Suppress repetitive alert during cooldown
                return

        self.hazard_state[obj_id] = {"time": current_time, "distance": new_dist}
        self.alert_queue.put((priority, current_time, payload))

    def reset_alert_state(self, object_id):
        """Clears the cooldown state for a specific alert (e.g., to announce path clear immediately after a hazard clears)."""
        if object_id in self.hazard_state:
            del self.hazard_state[object_id]

    def add_system_error(self, error_message):
        """Instantly injects a Priority 0 hardware/system failure alert."""
        payload = {
            "object_id": "sys_fatal",
            "priority": 0,
            "message": error_message,
            "hazard_type": "system",
            "distance_m": 0.0
        }
        self.alert_queue.put((0, time.time(), payload))

    def trigger_earcon(self, priority):
        """Fires an audible warning tone for critical priority alerts."""
        if priority <= 1 and HAVE_WINSOUND:
            try:
                winsound.Beep(1000, 200)
            except Exception:
                pass

    def _audio_worker(self):
        speaker = None
        if HAVE_SAPI:
            try:
                pythoncom.CoInitialize()
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                
                # Check for standard Indian English voice (Heera / Ravi / India)
                voices = speaker.GetVoices()
                for voice in voices:
                    desc = voice.GetDescription()
                    if "India" in desc or "Heera" in desc or "Ravi" in desc:
                        speaker.Voice = voice
                        break
            except Exception as e:
                print(f"[TTS WARNING] SAPI voice initialization failed: {e}. Falling back to console output.")
                speaker = None

        while True:
            priority, timestamp, payload = self.alert_queue.get()
            msg = payload.get("message", "")
            obj_id = payload.get("object_id", "")
            dist_m = payload.get("distance_m", 0.0)
            self.last_spoken_message = msg
            self.last_spoken_priority = priority

            # 1. Log TTS output to Console
            if priority == 0:
                print(f"\n[TTS AUDIO PRIORITY 0: SYSTEM FATAL] (1000Hz BEEP) >>> \"{msg}\"")
            elif priority == 1:
                print(f"\n[TTS AUDIO PRIORITY 1: URGENT HAZARD] (1000Hz BEEP) >>> \"{msg}\" (ID: {obj_id}, Dist: {dist_m:.1f}m)")
            elif priority == 2:
                print(f"\n[TTS AUDIO PRIORITY 2: DIRECTIONAL] >>> \"{msg}\" (Target: {obj_id})")
            else:
                print(f"\n[TTS AUDIO PRIORITY 3: GUIDANCE] >>> \"{msg}\"")

            # 2. Audible Sound Output
            if priority <= 1:
                self.trigger_earcon(priority)

            if speaker is not None:
                try:
                    if priority <= 1:
                        # Urgent rate: 1, flag 3 = SVSFlagsAsync | SVSFPurgeBeforeSpeak (purges current speech instantly)
                        speaker.Rate = 1
                        speaker.Speak(msg, 3)
                    else:
                        # Clear rate: -2, flag 1 = SVSFlagsAsync
                        speaker.Rate = -2
                        speaker.Speak(msg, 1)

                    # Keep worker alive while audio plays, but allow high-priority preemption
                    while speaker.Status.RunningState == 2:
                        time.sleep(0.04)
                        if not self.alert_queue.empty() and self.alert_queue.queue[0][0] <= 1:
                            break
                except Exception as e:
                    print(f"[TTS AUDIO ERROR] Speech playback failed: {e}")

            self.alert_queue.task_done()

