import cv2
import numpy as np
import librosa
import json
import os
import logging
import time
import base64
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DanceAnalyzer:
    def __init__(self):
        # Use YOLOv8 Nano Pose for speed/accuracy balance
        self.model_name = 'yolov8n-pose.pt'

    def _map_yolo_to_mp(self, keypoints, confs, image_shape):
        """
        Maps YOLOv8 keypoints (17 points) to MediaPipe Pose structure (33 points).
        Only maps critical landmarks for compatibility.

        YOLO Keypoints (COCO):
        0: Nose, 1: L Eye, 2: R Eye, 3: L Ear, 4: R Ear
        5: L Shoulder, 6: R Shoulder, 7: L Elbow, 8: R Elbow
        9: L Wrist, 10: R Wrist,
        11: L Hip, 12: R Hip, 13: L Knee, 14: R Knee
        15: L Ankle, 16: R Ankle

        MediaPipe Landmarks:
        0: Nose, 2: L Eye, 5: R Eye, 7: L Ear, 8: R Ear
        11: L Shoulder, 12: R Shoulder, 13: L Elbow, 14: R Elbow
        15: L Wrist, 16: R Wrist,
        23: L Hip, 24: R Hip, 25: L Knee, 26: R Knee
        27: L Ankle, 28: R Ankle
        """
        h, w = image_shape[:2]
        mp_landmarks = []

        # Initialize 33 empty landmarks
        for _ in range(33):
            mp_landmarks.append({"x": 0, "y": 0, "z": 0, "visibility": 0})

        # Mapping Dictionary: YOLO_Index -> MP_Index
        mapping = {
            0: 0,   # Nose
            1: 2,   # Left Eye
            2: 5,   # Right Eye
            3: 7,   # Left Ear
            4: 8,   # Right Ear
            5: 11,  # Left Shoulder
            6: 12,  # Right Shoulder
            7: 13,  # Left Elbow
            8: 14,  # Right Elbow
            9: 15,  # Left Wrist
            10: 16, # Right Wrist
            11: 23, # Left Hip
            12: 24, # Right Hip
            13: 25, # Left Knee
            14: 26, # Right Knee
            15: 27, # Left Ankle
            16: 28  # Right Ankle
        }

        for yolo_idx, mp_idx in mapping.items():
            if yolo_idx < len(keypoints):
                # YOLO coordinates are usually normalized if using .xyn,
                # but we need to check if we are passing normalized or pixel.
                # Here we assume we receive NORMALIZED coordinates (xyn).
                kp = keypoints[yolo_idx]
                conf = confs[yolo_idx] if confs is not None else 0.0

                mp_landmarks[mp_idx] = {
                    "x": float(kp[0]),
                    "y": float(kp[1]),
                    "z": 0.0, # 2D estimation only
                    "visibility": float(conf)
                }

        return mp_landmarks

    def find_candidates(self, file_path: str):
        """
        Scans the video for the first valid frame with people using YOLOv8.
        Returns the frame (base64) and a list of candidate bounding boxes.
        """
        model = YOLO(self.model_name)
        cap = cv2.VideoCapture(file_path)
        candidates = []
        frame_base64 = None

        try:
            frames_to_check = 30
            for _ in range(frames_to_check):
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO inference
                results = model(frame, verbose=False)

                valid_detections = []
                h, w, _ = frame.shape

                if results and len(results) > 0:
                    result = results[0]
                    # Check boxes
                    for i, box in enumerate(result.boxes):
                        cls = int(box.cls[0])
                        if cls != 0: # 0 is person in COCO
                            continue

                        # xywhn returns normalized center_x, center_y, w, h
                        # We need top_left_x, top_left_y, w, h for compatibility with frontend?
                        # Re-reading `find_candidates` in original code:
                        # Original used `bboxC.origin_x` etc. which is Top-Left.
                        # YOLO `xywhn` is Center-based. `xyxyn` is Top-Left/Bottom-Right.
                        # Let's use `xywhn` and convert or `xyxyn`.
                        # Let's use box.xyxyn for normalized coords.

                        b = box.xyxyn[0].cpu().numpy() # x1, y1, x2, y2 normalized
                        x1, y1, x2, y2 = b
                        width_n = x2 - x1
                        height_n = y2 - y1

                        bbox = {
                            "id": i,
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(width_n),
                            "height": float(height_n),
                            "score": float(box.conf[0])
                        }
                        valid_detections.append(bbox)

                if valid_detections:
                    candidates = valid_detections
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    break

        finally:
            cap.release()
            # No explicit close needed for YOLO object, but good practice to clear if possible
            del model

        return {"image": frame_base64, "candidates": candidates}

    def analyze_video(self, file_path: str, tracking_config: dict = None):
        """
        Generator function that yields progress updates and finally the result.
        Uses YOLOv8-Pose with ByteTrack for multi-person tracking.
        """
        logger.info(f"Starting analysis for {file_path}")
        yield {"status": "starting", "progress": 0, "message": "Initializing analysis..."}

        start_time = time.time()

        # Load Model
        try:
            model = YOLO(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            yield {"status": "error", "message": "Failed to load AI model"}
            return

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
             yield {"status": "error", "progress": 0, "message": "Could not open video file"}
             return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pose_data = []
        frames_processed = 0

        logger.info(f"Video info: FPS={fps}, Frames={frame_count}")
        yield {"status": "processing_video", "progress": 0, "message": f"Starting video processing ({frame_count} frames)..."}

        # --- Tracking State ---
        target_track_id = None
        excluded_ids = set() # IDs to exclude (accompanists, etc.)
        last_target_pos = None # Last known target position (cx, cy)

        # Store history for recovery logic: {track_id: [list of (cx, cy) positions]}
        track_histories = {}
        # Store skeletal signature: {track_id: ratio}
        # Ratio = (LeftHip-RightHip dist) / (MidHip-MidShoulder dist) approx?
        # Simpler: Just height/width of bounding box?
        target_signature = None

        pose_start = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO Tracking
                # persist=True is crucial for ID consistency
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                current_frame_data = {
                    "frame": frames_processed,
                    "timestamp": frames_processed / fps if fps else 0,
                    "people": [],
                    # Legacy support for older metrics calculation (will point to target)
                    "landmarks": []
                }

                if results and results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    keypoints = results[0].keypoints

                    # Convert tensors to numpy
                    track_ids = boxes.id.int().cpu().numpy().tolist()
                    cls_ids = boxes.cls.int().cpu().numpy().tolist()
                    boxes_xywhn = boxes.xywhn.cpu().numpy() # Center x,y, w, h normalized

                    # Normalized keypoints for mapping
                    kpts_xyn = keypoints.xyn.cpu().numpy()
                    kpts_conf = keypoints.conf.cpu().numpy()

                    # 1. ID Selection (First Frame / Initialization)
                    if target_track_id is None and tracking_config and "target_bbox" in tracking_config:
                        tb = tracking_config["target_bbox"]
                        # Target BBox Center
                        if isinstance(tb, dict):
                            tx = tb['x'] + tb['width'] / 2
                            ty = tb['y'] + tb['height'] / 2
                        else:
                            # Assume list [x, y, w, h]
                            tx = tb[0] + tb[2] / 2
                            ty = tb[1] + tb[3] / 2

                        best_dist = float('inf')
                        best_id = -1
                        best_idx = -1

                        for i, tid in enumerate(track_ids):
                            if cls_ids[i] != 0: continue # Only persons

                            bx, by, bw, bh = boxes_xywhn[i]
                            dist = np.sqrt((tx - bx)**2 + (ty - by)**2)

                            if dist < best_dist:
                                best_dist = dist
                                best_id = tid
                                best_idx = i

                        if best_id != -1:
                            target_track_id = best_id
                            # Store initial signature (simple aspect ratio for now)
                            _, _, w_b, h_b = boxes_xywhn[best_idx]
                            target_signature = w_b / h_b if h_b > 0 else 0

                            # Initial last position
                            bx, by, _, _ = boxes_xywhn[best_idx]
                            last_target_pos = (bx, by)

                            # Blacklist other IDs present in this frame
                            for tid in track_ids:
                                if tid != target_track_id:
                                    excluded_ids.add(tid)

                            logger.info(f"Locked on Track ID: {target_track_id}")
                            logger.info(f"Excluded IDs (Accompanists): {excluded_ids}")

                    # 2. Update Histories (for all people)
                    current_track_indices = {}
                    for i, tid in enumerate(track_ids):
                        if cls_ids[i] != 0: continue

                        if tid not in track_histories:
                            track_histories[tid] = []

                        cx, cy, _, _ = boxes_xywhn[i]
                        track_histories[tid].append((cx, cy))
                        # Keep history short (e.g., 30 frames)
                        if len(track_histories[tid]) > 30:
                            track_histories[tid].pop(0)

                        current_track_indices[tid] = i

                    # 3. Target Retrieval or Recovery
                    target_idx = -1

                    if target_track_id in current_track_indices:
                        target_idx = current_track_indices[target_track_id]
                        # Update last known position
                        bx, by, _, _ = boxes_xywhn[target_idx]
                        last_target_pos = (bx, by)
                    else:
                        # TARGET LOST - RECOVERY LOGIC
                        if target_track_id is not None:
                            best_recovery_score = -1
                            best_recovery_id = None

                            logger.info(f"Target {target_track_id} lost. Attempting recovery...")

                            for tid, idx in current_track_indices.items():
                                # CRITERIA 1: Not in Excluded List
                                if tid in excluded_ids:
                                    continue

                                # CRITERIA 2: Proximity to last known position (60%)
                                proximity_score = 0
                                if last_target_pos:
                                    lx, ly = last_target_pos
                                    cx, cy, _, _ = boxes_xywhn[idx]
                                    dist = np.sqrt((lx - cx)**2 + (ly - cy)**2)
                                    # Normalize: distance > 0.5 (half screen) implies 0 score
                                    proximity_score = max(0, 1.0 - dist * 2)

                                # CRITERIA 3: Movement Intensity (40%)
                                # Yasugi-bushi dancer moves more than static people
                                hist = track_histories.get(tid, [])
                                movement_score = 0
                                if len(hist) > 5:
                                    hist_arr = np.array(hist)
                                    # Calculate total path length (velocity)
                                    velocity = np.sum(np.sqrt(np.diff(hist_arr[:,0])**2 + np.diff(hist_arr[:,1])**2))
                                    movement_score = min(velocity * 10, 1.0) # Normalize loosely

                                # Combined Score (Weighted)
                                # Distance (0.6) + Movement (0.4)
                                total_score = 0.6 * proximity_score + 0.4 * movement_score

                                if total_score > best_recovery_score:
                                    best_recovery_score = total_score
                                    best_recovery_id = tid

                            # Threshold to accept switch
                            if best_recovery_id is not None and best_recovery_score > 0.3:
                                logger.info(f"Target lost. Switching from {target_track_id} to {best_recovery_id} (Score: {best_recovery_score:.2f})")
                                target_track_id = best_recovery_id
                                target_idx = current_track_indices[best_recovery_id]
                                # Update position
                                bx, by, _, _ = boxes_xywhn[target_idx]
                                last_target_pos = (bx, by)

                    # 4. Extract Data for ALL people
                    for i, tid in enumerate(track_ids):
                         if cls_ids[i] != 0: continue

                         # Extract keypoints
                         kp = kpts_xyn[i]
                         cf = kpts_conf[i]

                         # Map to MediaPipe format
                         mp_landmarks = self._map_yolo_to_mp(kp, cf, (height, width))

                         is_target = (tid == target_track_id)

                         person_data = {
                             "id": tid,
                             "is_target": is_target,
                             "landmarks": mp_landmarks
                         }
                         current_frame_data["people"].append(person_data)

                         # Populate legacy field if this is the target
                         if is_target:
                             current_frame_data["landmarks"] = mp_landmarks

                pose_data.append(current_frame_data)
                frames_processed += 1

                if frames_processed % 10 == 0 or frames_processed == frame_count:
                    progress = (frames_processed / max(frame_count, 1)) * 0.7
                    yield {
                        "status": "processing_video",
                        "progress": round(progress, 3),
                        "message": f"Processing video frame {frames_processed}/{frame_count}"
                    }

        finally:
            cap.release()
            del model

        pose_end = time.time()
        logger.info(f"Pose analysis took {pose_end - pose_start:.2f}s")

        # 2. Audio Analysis with Librosa (Unchanged)
        yield {"status": "processing_audio", "progress": 0.7, "message": "Analyzing audio..."}
        audio_start = time.time()

        audio_data = {}
        try:
            y, sr = librosa.load(file_path)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            duration = librosa.get_duration(y=y, sr=sr)

            audio_data = {
                "tempo": float(tempo),
                "onset_env": onset_env.tolist()[::10],
                "duration": duration,
                "sr": sr
            }
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            audio_data = {"error": str(e)}

        audio_end = time.time()
        logger.info(f"Audio analysis took {audio_end - audio_start:.2f}s")
        yield {"status": "processing_audio", "progress": 0.9, "message": "Audio analysis complete"}

        # 3. Calculate Metrics (Unchanged)
        yield {"status": "calculating_metrics", "progress": 0.95, "message": "Calculating metrics..."}
        metrics = self.calculate_metrics(pose_data, audio_data)

        final_result = {
            "pose_data": pose_data,
            "audio_data": audio_data,
            "metrics": metrics
        }

        total_time = time.time() - start_time
        logger.info(f"Total analysis took {total_time:.2f}s")

        yield {"status": "complete", "progress": 1.0, "message": "Analysis complete", "result": final_result}

    def calculate_metrics(self, pose_data, audio_data):
        if not pose_data:
            return {}

        # Helper to get target landmarks from frame
        def get_target_landmarks(frame_data):
            # Prefer new 'people' list
            if "people" in frame_data:
                for p in frame_data["people"]:
                    if p.get("is_target"):
                        return p["landmarks"]
            # Fallback to legacy
            return frame_data.get("landmarks", [])

        # 1. Koshi (MidHip Stability)
        # MidHip is roughly average of Left Hip (23) and Right Hip (24)
        mid_hip_y_variances = []
        for frame in pose_data:
            landmarks = get_target_landmarks(frame)
            # Check for non-empty landmarks (YOLO might miss target in some frames)
            # Check if index 23/24 exist and are not zero (or handle zero)
            if len(landmarks) > 24:
                left_hip = landmarks[23]
                right_hip = landmarks[24]

                # Check visibility/confidence > 0 to ensure valid detection
                if left_hip['visibility'] > 0.3 and right_hip['visibility'] > 0.3:
                    mid_hip_y = (left_hip['y'] + right_hip['y']) / 2
                    mid_hip_y_variances.append(mid_hip_y)

        stability_score = 0.0
        if mid_hip_y_variances:
            variance = np.var(mid_hip_y_variances)
            stability_score = max(0, 1.0 - (variance * 10))

        # 2. Kire (Jerk of Hands)
        # Calculate jerk for hands (15, 16)
        hand_velocities = []
        for i in range(1, len(pose_data)):
            curr = get_target_landmarks(pose_data[i])
            prev = get_target_landmarks(pose_data[i-1])

            if not curr or not prev or len(curr) <= 16 or len(prev) <= 16:
                continue

            # Check visibility
            if (curr[15]['visibility'] > 0.3 and prev[15]['visibility'] > 0.3 and
                curr[16]['visibility'] > 0.3 and prev[16]['visibility'] > 0.3):

                # Left wrist (15) dist
                d_left = np.sqrt((curr[15]['x']-prev[15]['x'])**2 + (curr[15]['y']-prev[15]['y'])**2)
                # Right wrist (16) dist
                d_right = np.sqrt((curr[16]['x']-prev[16]['x'])**2 + (curr[16]['y']-prev[16]['y'])**2)
                hand_velocities.append((d_left + d_right) / 2)

        dynamism_score = 0.0
        if hand_velocities:
            avg_vel = np.mean(hand_velocities)
            max_vel = np.max(hand_velocities)
            if avg_vel > 0:
                dynamism_score = min(1.0, (max_vel / avg_vel) / 5.0)

        # 3. Ma (Rhythm Harmony / Pause)
        rhythm_score = 0.5
        if hand_velocities:
            threshold = np.mean(hand_velocities) * 0.2
            pause_frames = sum(1 for v in hand_velocities if v < threshold)
            pause_ratio = pause_frames / len(hand_velocities)
            rhythm_score = 1.0 - abs(pause_ratio - 0.15) * 2
            rhythm_score = max(0, min(1.0, rhythm_score))

        return {
            "stability_score": float(stability_score),
            "rhythm_score": float(rhythm_score),
            "dynamism_score": float(dynamism_score)
        }
