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

# --- New Tracking Components ---

class FeatureExtractor:
    @staticmethod
    def extract_hsv_histogram(image: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extracts HSV color histogram from the bbox region.
        bbox: (x1, y1, x2, y2) in pixels (integers).
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calculate Histogram (Hue and Saturation only to be robust to brightness)
        # 30 bins for Hue, 32 bins for Saturation
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    @staticmethod
    def compare_histograms(hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0.0
        # Correlation method
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


class PoseValidator:
    @staticmethod
    def validate_pose(keypoints, confs, prev_keypoints=None):
        """
        Validates the pose based on anatomical constraints.
        keypoints: shape (17, 2) normalized or pixel coordinates.
        confs: shape (17,) confidence scores.
        prev_keypoints: shape (17, 2) from previous frame (optional).

        Returns: (is_valid, reason_str)
        """
        # Indices (COCO)
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16

        # 1. Vertical Order Check (Head < Shoulders < Hips < Knees < Ankles)
        # We use average Y for left/right pairs to be robust
        def get_y(indices):
            valid_ys = [keypoints[i][1] for i in indices if i < len(confs) and confs[i] > 0.3]
            return np.mean(valid_ys) if valid_ys else None

        y_head = get_y([NOSE])
        y_shoulder = get_y([L_SHOULDER, R_SHOULDER])
        y_hip = get_y([L_HIP, R_HIP])
        y_knee = get_y([L_KNEE, R_KNEE])
        y_ankle = get_y([L_ANKLE, R_ANKLE])

        # Yasuki-Bushi: Head must be above hips (y_head < y_hip)
        if y_head is not None and y_hip is not None:
            if y_head > y_hip:
                return False, "Head below hips (Inverted Pose)"

        # General Structure (Allow some flexibility, e.g. bent knees, but Hips should be below Shoulders)
        if y_shoulder is not None and y_hip is not None:
            if y_shoulder > y_hip:
                 return False, "Shoulders below hips"

        if y_hip is not None and y_knee is not None:
            if y_hip > y_knee:
                return False, "Hips below knees"

        if y_knee is not None and y_ankle is not None:
            if y_knee > y_ankle:
                return False, "Knees below ankles"

        # 2. Segment Length Continuity
        if prev_keypoints is not None:
            segments = [
                (5, 7), (6, 8),   # Shoulder-Elbow
                (7, 9), (8, 10),  # Elbow-Wrist
                (5, 6),           # Shoulder-Shoulder
                (11, 13), (12, 14) # Hip-Knee
            ]

            for i1, i2 in segments:
                if i1 >= len(confs) or i2 >= len(confs): continue
                # Check if points are valid in both frames
                if confs[i1] > 0.3 and confs[i2] > 0.3:
                    # Calculate current length
                    curr_dist = np.linalg.norm(keypoints[i1] - keypoints[i2])

                    # Calculate prev length
                    prev_p1 = prev_keypoints[i1]
                    prev_p2 = prev_keypoints[i2]
                    prev_dist = np.linalg.norm(prev_p1 - prev_p2)

                    if prev_dist > 0:
                        change = abs(curr_dist - prev_dist) / prev_dist
                        if change > 0.6:
                            return False, f"Segment {i1}-{i2} length changed by {change:.2f}"

        return True, "Valid"

    @staticmethod
    def check_connection_quality(keypoints, confs):
        """
        Returns a score boost if the main body line (Shoulder-Hip-Knee-Ankle) is solid.
        """
        score = 0
        if len(confs) <= 16: return 0
        # Check Left Line: 5-11-13-15
        if confs[5] > 0.5 and confs[11] > 0.5 and confs[13] > 0.5 and confs[15] > 0.5:
            score += 1
        # Check Right Line: 6-12-14-16
        if confs[6] > 0.5 and confs[12] > 0.5 and confs[14] > 0.5 and confs[16] > 0.5:
            score += 1
        return score


class PersonProfile:
    def __init__(self, track_id, initial_pos_norm, initial_hist, aspect_ratio):
        self.id = track_id
        self.positions = [initial_pos_norm] # List of (cx, cy) normalized
        self.color_hist = initial_hist
        self.aspect_ratio = aspect_ratio
        self.last_keypoints = None
        self.prev_keypoints = None # Frame N-1
        self.last_keypoint_confs = None

        self.total_movement = 0.0
        self.last_seen_frame = 0
        self.is_static = False

        # Parameters
        self.history_len = 60 # Frames to keep for static check

    def update(self, pos_norm, hist, aspect_ratio, frame_idx, keypoints=None, keypoint_confs=None):
        self.last_seen_frame = frame_idx
        self.aspect_ratio = aspect_ratio # Update latest aspect ratio
        if keypoints is not None:
            self.prev_keypoints = self.last_keypoints
            self.last_keypoints = keypoints
            self.last_keypoint_confs = keypoint_confs

        # Update Movement
        prev_pos = self.positions[-1]
        dist = np.linalg.norm(np.array(pos_norm) - np.array(prev_pos))
        self.total_movement += dist

        # Update Position History
        self.positions.append(pos_norm)
        if len(self.positions) > self.history_len:
            self.positions.pop(0)

        # Update Color Histogram (Exponential Moving Average)
        if hist is not None:
            if self.color_hist is None:
                self.color_hist = hist
            else:
                # Weighted average: 90% old, 10% new
                self.color_hist = cv2.addWeighted(self.color_hist, 0.9, hist, 0.1, 0)

        # Check Static State
        self._check_static()

    def _check_static(self):
        if len(self.positions) < self.history_len:
            self.is_static = False
            return

        # Calculate variance of positions
        pts = np.array(self.positions)
        # pts is (N, 2)
        var = np.var(pts, axis=0) # [var_x, var_y]

        # Threshold for "extremely small variance"
        # Since coords are normalized 0-1, 1e-5 is very small (approx 3-4 pixels in 1080p)
        if np.mean(var) < 1e-5:
            self.is_static = True
        else:
            self.is_static = False


class TrackerManager:
    def __init__(self, initial_target_bbox_norm, image_shape):
        """
        initial_target_bbox_norm: dict {x, y, width, height} or list [x, y, w, h] (normalized)
        """
        self.h, self.w = image_shape[:2]

        # State
        self.profiles = {} # id -> PersonProfile
        self.target_id = None

        # Exclusion List: List of PersonProfile (snapshots of lost non-target tracks)
        self.historical_exclusions = []

        # Target Tracking State
        self.target_last_pos_norm = None
        self.target_last_bbox_pixel = None # For sit-tight extraction
        self.target_last_hist = None

        # Initialize target search parameters
        self._parse_initial_target(initial_target_bbox_norm)

    def _parse_initial_target(self, bbox):
        if bbox is None:
            self.initial_target_center = (0.5, 0.5) # Default?
            return

        if isinstance(bbox, dict):
            cx = bbox['x'] + bbox['width'] / 2
            cy = bbox['y'] + bbox['height'] / 2
        else:
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
        self.initial_target_center = (cx, cy)

    def update(self, frame, detections, frame_idx):
        """
        detections: List of dicts {
            'id': int,
            'bbox_norm': [x1, y1, x2, y2],
            'bbox_pixel': [x1, y1, x2, y2]
        }
        Returns: current_target_id, is_virtual_tracking
        """
        current_ids = set()

        # 1. Update/Create Profiles
        for det in detections:
            tid = det['id']
            bbox_norm = det['bbox_norm'] # x1,y1,x2,y2
            bbox_pix = det['bbox_pixel']

            # Calculate Center and Aspect Ratio
            w_n = bbox_norm[2] - bbox_norm[0]
            h_n = bbox_norm[3] - bbox_norm[1]
            cx, cy = bbox_norm[0] + w_n/2, bbox_norm[1] + h_n/2
            ar = w_n / h_n if h_n > 0 else 0

            # Extract Features
            hist = FeatureExtractor.extract_hsv_histogram(frame, bbox_pix)

            current_ids.add(tid)

            if tid not in self.profiles:
                # New ID. Check if this matches a known exclusion.
                if self._matches_exclusion((cx, cy), ar, hist):
                    logger.info(f"ID {tid} matches exclusion list. Marking as excluded.")
                    # We create the profile but can flag it if we added a flag,
                    # or just rely on 'historical_exclusions' check during recovery.
                    # Ideally we want to prevent this ID from BECOMING the target.
                    pass

                self.profiles[tid] = PersonProfile(tid, (cx, cy), hist, ar)
            else:
                kp = det.get('keypoints')
                kp_conf = det.get('keypoint_confs')
                self.profiles[tid].update((cx, cy), hist, ar, frame_idx, keypoints=kp, keypoint_confs=kp_conf)

        # Handle Lost Profiles: Move to historical exclusions
        # (This helps re-identifying them if they get a new ID)
        for pid, prof in list(self.profiles.items()):
            if pid not in current_ids and pid != self.target_id:
                # This non-target person disappeared. Snapshot their stats.
                self.historical_exclusions.append({
                    'pos': prof.positions[-1],
                    'ar': prof.aspect_ratio,
                    'color': prof.color_hist
                })
                logger.info(f"ID {pid} lost. Added to exclusion list.")
                # Remove from active profiles to avoid reprocessing
                del self.profiles[pid]

        # 2. Initialize Target if first run
        if self.target_id is None:
            self._initialize_target(detections)
            return self.target_id, False

        # 3. Target Tracking Logic
        if self.target_id in current_ids:
            # Target FOUND

            # --- Hard Gating & Validation ---
            # Even if ID matches, verify it's physically possible
            valid_match = True

            # Find the detection
            target_det = next((d for d in detections if d['id'] == self.target_id), None)

            if target_det and self.target_last_bbox_pixel is not None:
                # A. Spatial Hard Gating
                prev_w = self.target_last_bbox_pixel[2] - self.target_last_bbox_pixel[0]
                prev_h = self.target_last_bbox_pixel[3] - self.target_last_bbox_pixel[1]

                curr_cx, curr_cy = target_det['bbox_pixel'][0] + (target_det['bbox_pixel'][2]-target_det['bbox_pixel'][0])/2, \
                                   target_det['bbox_pixel'][1] + (target_det['bbox_pixel'][3]-target_det['bbox_pixel'][1])/2

                prev_cx, prev_cy = self.target_last_bbox_pixel[0] + prev_w/2, self.target_last_bbox_pixel[1] + prev_h/2

                dx = abs(curr_cx - prev_cx)
                dy = abs(curr_cy - prev_cy)

                if dy > 0.7 * prev_h or dx > 1.2 * prev_w:
                    logger.warning(f"Hard Gating: Rejected target {self.target_id} (Moved too fast: dx={dx:.1f}, dy={dy:.1f})")
                    valid_match = False

                # B. Anatomical Validation
                if valid_match and 'keypoints' in target_det:
                    kp = target_det['keypoints']
                    conf = target_det['keypoint_confs']
                    # We use prev_keypoints because last_keypoints was just updated with current frame data
                    prev_kp = self.profiles[self.target_id].prev_keypoints

                    is_valid_pose, msg = PoseValidator.validate_pose(kp, conf, prev_kp)
                    if not is_valid_pose:
                        logger.warning(f"Anatomical Check: Rejected target {self.target_id} ({msg})")
                        valid_match = False

            if valid_match:
                self._update_target_state(detections)
                return self.target_id, False
            else:
                # Treat as LOST despite ID match
                # We do NOT return here, so it falls through to Recovery/Sit-Tight logic
                pass

        # TARGET LOST -> Recovery or Sit-Tight

            # A. Try Recovery First (Switch to better candidate)
            # If a new ID appears that looks like the target, we prefer switching to it
            # rather than assuming the target is "invisible" (Sit-Tight).
            best_id = self._find_recovery_candidate(current_ids)
            if best_id is not None:
                logger.info(f"Recovered target: Switched from {self.target_id} to {best_id}")
                self.target_id = best_id
                self._update_target_state(detections)
                return self.target_id, False

            # B. Sit-Tight Logic (If no good candidate found)
            if self.target_last_bbox_pixel is not None:
                # Check color at last position
                current_patch_hist = FeatureExtractor.extract_hsv_histogram(frame, self.target_last_bbox_pixel)
                similarity = FeatureExtractor.compare_histograms(current_patch_hist, self.target_last_hist)

                # Threshold for "Same Color"
                if similarity > 0.6:
                    # Assume target is still there
                    return self.target_id, True # True = Virtual Tracking

            # Truly Lost
            return self.target_id, True

    def _update_target_state(self, detections):
        # Find detection for target_id
        for det in detections:
            if det['id'] == self.target_id:
                # Update persistent target state
                bbox_norm = det['bbox_norm']
                w_n = bbox_norm[2] - bbox_norm[0]
                h_n = bbox_norm[3] - bbox_norm[1]
                cx, cy = bbox_norm[0] + w_n/2, bbox_norm[1] + h_n/2

                self.target_last_pos_norm = (cx, cy)
                self.target_last_bbox_pixel = det['bbox_pixel']
                self.target_last_hist = self.profiles[self.target_id].color_hist
                break

    def _initialize_target(self, detections):
        # Find closest to initial_target_center
        best_dist = float('inf')
        best_id = None

        tx, ty = self.initial_target_center

        for det in detections:
            bbox_norm = det['bbox_norm']
            w_n = bbox_norm[2] - bbox_norm[0]
            h_n = bbox_norm[3] - bbox_norm[1]
            cx, cy = bbox_norm[0] + w_n/2, bbox_norm[1] + h_n/2

            dist = np.sqrt((tx-cx)**2 + (ty-cy)**2)
            if dist < best_dist:
                best_dist = dist
                best_id = det['id']

        if best_id is not None:
            self.target_id = best_id
            self._update_target_state(detections)
            logger.info(f"Tracker initialized. Target ID: {self.target_id}")

    def _matches_exclusion(self, pos, ar, hist):
        # Check against historical exclusions
        for exc in self.historical_exclusions:
            d_pos = np.linalg.norm(np.array(pos) - np.array(exc['pos']))
            d_ar = abs(ar - exc['ar'])
            sim_color = FeatureExtractor.compare_histograms(hist, exc['color'])

            if d_pos < 0.1 and sim_color > 0.8: # Strict check
                return True
        return False

    def _find_recovery_candidate(self, current_ids):
        best_score = -1
        best_cand = None

        for tid in current_ids:
            if tid == self.target_id: continue

            prof = self.profiles[tid]

            # 1. Background/Static Filter
            if prof.is_static:
                continue

            # 2. Exclusion Check
            if self._matches_exclusion(prof.positions[-1], prof.aspect_ratio, prof.color_hist):
                continue

            # 3. Score Calculation
            # "Proximity (60%) + Color (40%)" - NO Aspect Ratio
            if self.target_last_pos_norm is not None and self.target_last_hist is not None:
                dist = np.linalg.norm(np.array(prof.positions[-1]) - np.array(self.target_last_pos_norm))
                color_sim = FeatureExtractor.compare_histograms(prof.color_hist, self.target_last_hist)

                # Norm Distance 0-1.
                dist_score = max(0, 1.0 - (dist * 2)) # If dist > 0.5, score is 0

                score = 0.6 * dist_score + 0.4 * color_sim

                # Boost for good structure
                if prof.last_keypoints is not None and prof.last_keypoint_confs is not None:
                    conn_score = PoseValidator.check_connection_quality(prof.last_keypoints, prof.last_keypoint_confs)
                    score += (conn_score * 0.1) # Boost up to 0.2

                if score > best_score:
                    best_score = score
                    best_cand = tid

        if best_score > 0.4: # Threshold
            return best_cand
        return None

# --- Main Class ---

class DanceAnalyzer:
    def __init__(self):
        # Use YOLOv8 Nano Pose for speed/accuracy balance
        self.model_name = 'yolov8n-pose.pt'

    def _map_yolo_to_mp(self, keypoints, confs, image_shape):
        """
        Maps YOLOv8 keypoints (17 points) to MediaPipe Pose structure (33 points).
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
                kp = keypoints[yolo_idx]
                conf = confs[yolo_idx] if confs is not None else 0.0

                mp_landmarks[mp_idx] = {
                    "x": float(kp[0]),
                    "y": float(kp[1]),
                    "z": 0.0,
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

                if results and len(results) > 0:
                    result = results[0]
                    # Check boxes
                    for i, box in enumerate(result.boxes):
                        cls = int(box.cls[0])
                        if cls != 0: # 0 is person in COCO
                            continue

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
            del model

        return {"image": frame_base64, "candidates": candidates}

    def analyze_video(self, file_path: str, tracking_config: dict = None):
        """
        Generator function that yields progress updates and finally the result.
        Uses YOLOv8-Pose with ByteTrack for multi-person tracking and Custom TrackerManager.
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

        # --- Tracker Initialization ---
        target_bbox = None
        if tracking_config and "target_bbox" in tracking_config:
            target_bbox = tracking_config["target_bbox"]

        tm = TrackerManager(target_bbox, (height, width))

        pose_start = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO Tracking
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                current_frame_data = {
                    "frame": frames_processed,
                    "timestamp": frames_processed / fps if fps else 0,
                    "people": [],
                    "landmarks": []
                }

                # 1. Parse Detections
                detections = []
                # Map from id to keypoints/confs for later retrieval
                detection_map = {}

                if results and results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    keypoints = results[0].keypoints

                    track_ids = boxes.id.int().cpu().numpy().tolist()
                    cls_ids = boxes.cls.int().cpu().numpy().tolist()

                    # Boxes
                    boxes_xyxyn = boxes.xyxyn.cpu().numpy() # x1, y1, x2, y2 normalized
                    boxes_xyxy = boxes.xyxy.cpu().numpy() # pixel coords

                    # Keypoints
                    kpts_xyn = keypoints.xyn.cpu().numpy()
                    kpts_conf = keypoints.conf.cpu().numpy()

                    for i, tid in enumerate(track_ids):
                        if cls_ids[i] != 0: continue

                        det = {
                            'id': tid,
                            'bbox_norm': boxes_xyxyn[i],
                            'bbox_pixel': boxes_xyxy[i],
                            'keypoints': kpts_xyn[i],
                            'keypoint_confs': kpts_conf[i],
                            'aspect_ratio': 0 # Calculated in TM
                        }
                        detections.append(det)
                        detection_map[tid] = {
                            'kp': kpts_xyn[i],
                            'conf': kpts_conf[i]
                        }

                # 2. Update Tracker Logic
                target_id, is_virtual = tm.update(frame, detections, frames_processed)

                # 3. Construct Output
                # Even if detection_map is empty (no YOLO detections), we might have virtual tracking

                # If target is virtual, we reuse last known position?
                # The frontend expects "people" array.

                # Iterate over ALL active tracks (or just the detected ones?)
                # We should output what YOLO sees, plus maybe the virtual target if not seen.

                for tid, data in detection_map.items():
                    is_target = (tid == target_id)

                    mp_landmarks = self._map_yolo_to_mp(data['kp'], data['conf'], (height, width))

                    person_data = {
                        "id": tid,
                        "is_target": is_target,
                        "landmarks": mp_landmarks
                    }
                    current_frame_data["people"].append(person_data)

                    if is_target:
                        current_frame_data["landmarks"] = mp_landmarks

                # If Virtual Tracking is active (Target ID not in detection_map),
                # We can optionally output a placeholder to indicate "Target is here but occluded/undetected"
                # But we don't have new keypoints.
                # For this assignment, "Maintain previous coordinate" in logic is done.
                # If we want to show it on frontend, we'd need to send something.
                # However, without keypoints, `landmarks` would be empty or stale.
                # Let's trust the requirement "Virtual tracking" was mainly to prevent switching to WRONG target.

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

        # 2. Audio Analysis
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
        yield {"status": "processing_audio", "progress": 0.9, "message": "Audio analysis complete"}

        # 3. Calculate Metrics
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

        def get_target_landmarks(frame_data):
            if "people" in frame_data:
                for p in frame_data["people"]:
                    if p.get("is_target"):
                        return p["landmarks"]
            return frame_data.get("landmarks", [])

        # 1. Koshi (MidHip Stability)
        mid_hip_y_variances = []
        for frame in pose_data:
            landmarks = get_target_landmarks(frame)
            if len(landmarks) > 24:
                left_hip = landmarks[23]
                right_hip = landmarks[24]

                if left_hip['visibility'] > 0.3 and right_hip['visibility'] > 0.3:
                    mid_hip_y = (left_hip['y'] + right_hip['y']) / 2
                    mid_hip_y_variances.append(mid_hip_y)

        stability_score = 0.0
        if mid_hip_y_variances:
            variance = np.var(mid_hip_y_variances)
            stability_score = max(0, 1.0 - (variance * 10))

        # 2. Kire (Jerk of Hands)
        hand_velocities = []
        for i in range(1, len(pose_data)):
            curr = get_target_landmarks(pose_data[i])
            prev = get_target_landmarks(pose_data[i-1])

            if not curr or not prev or len(curr) <= 16 or len(prev) <= 16:
                continue

            if (curr[15]['visibility'] > 0.3 and prev[15]['visibility'] > 0.3 and
                curr[16]['visibility'] > 0.3 and prev[16]['visibility'] > 0.3):

                d_left = np.sqrt((curr[15]['x']-prev[15]['x'])**2 + (curr[15]['y']-prev[15]['y'])**2)
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
