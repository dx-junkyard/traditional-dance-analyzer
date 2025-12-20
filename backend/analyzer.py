import cv2
import mediapipe as mp
import numpy as np
import librosa
import json
import os
import logging
import time
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_torso_roi(image, landmarks):
    """
    Extracts the torso ROI based on shoulder (11, 12) and hip (23, 24) landmarks.
    Landmarks are normalized [0, 1].
    """
    h, w, _ = image.shape

    # Get coordinates
    pts = [
        landmarks[11], # Left Shoulder
        landmarks[12], # Right Shoulder
        landmarks[23], # Left Hip
        landmarks[24]  # Right Hip
    ]

    # Convert to pixel coords
    pixel_pts = []
    for pt in pts:
        pixel_pts.append([int(pt.x * w), int(pt.y * h)])

    pixel_pts = np.array(pixel_pts)

    # Get bounding rect of these points
    x, y, bw, bh = cv2.boundingRect(pixel_pts)

    # Add slight padding
    pad_x = int(bw * 0.1)
    pad_y = int(bh * 0.1)

    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(w - x, bw + 2 * pad_x)
    bh = min(h - y, bh + 2 * pad_y)

    if bw <= 0 or bh <= 0:
        return None

    return image[y:y+bh, x:x+bw]

def calculate_color_histogram(image):
    """
    Calculates normalized 2D HSV histogram (Hue and Saturation).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 30 bins for Hue, 32 for Saturation
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

class DanceAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

        # Check for TFLite model, use relative path or download
        model_path = os.path.join(os.getcwd(), 'efficientdet_lite0.tflite')

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            max_results=5,
            score_threshold=0.3
        )
        self.detector = ObjectDetector.create_from_options(options)

    def find_candidates(self, file_path: str):
        """
        Scans the video for the first valid frame with people.
        Returns the frame (base64) and a list of candidate bounding boxes.
        """
        cap = cv2.VideoCapture(file_path)
        candidates = []
        frame_base64 = None

        frames_to_check = 30 # Check first 30 frames for a good shot

        for _ in range(frames_to_check):
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            results = self.detector.detect(mp_image)

            if results.detections:
                h, w, _ = frame.shape
                valid_detections = []
                for i, detection in enumerate(results.detections):
                    # For efficientdet_lite0, class 'person' usually has index 0 if labeled,
                    # but detection.categories[0].category_name should be checked.
                    # Or we just accept all objects (usually people in dance videos) or filter by 'person'.

                    category = detection.categories[0]
                    if category.category_name != 'person':
                        continue

                    bboxC = detection.bounding_box
                    # mp.tasks returns pixel coordinates in bounding_box (origin_x, origin_y, width, height)
                    # We normalize it for the frontend
                    bbox = {
                        "id": i,
                        "x": bboxC.origin_x / w,
                        "y": bboxC.origin_y / h,
                        "width": bboxC.width / w,
                        "height": bboxC.height / h,
                        "score": category.score
                    }
                    valid_detections.append(bbox)

                if valid_detections:
                    candidates = valid_detections
                    # Encode frame to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    break # Found candidates

        cap.release()
        return {"image": frame_base64, "candidates": candidates}

    def analyze_video(self, file_path: str, tracking_config: dict = None):
        """
        Generator function that yields progress updates and finally the result.
        tracking_config: {"target_bbox": [x, y, w, h]} normalized
        """
        logger.info(f"Starting analysis for {file_path}")
        yield {"status": "starting", "progress": 0, "message": "Initializing analysis..."}

        start_time = time.time()

        # 1. Pose Analysis with MediaPipe
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
             yield {"status": "error", "progress": 0, "message": "Could not open video file"}
             return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pose_data = []
        frames_processed = 0

        logger.info(f"Video info: FPS={fps}, Frames={frame_count}")
        yield {"status": "processing_video", "progress": 0, "message": f"Starting video processing ({frame_count} frames)..."}

        # Tracking State
        target_hist = None
        last_bbox = None # [x, y, w, h] normalized

        if tracking_config and "target_bbox" in tracking_config:
            last_bbox = tracking_config["target_bbox"]
            # Convert dict/list if needed. Assuming list [x, y, w, h]
            if isinstance(last_bbox, dict):
                last_bbox = [last_bbox['x'], last_bbox['y'], last_bbox['width'], last_bbox['height']]

        pose_start = time.time()

        # We need to initialize the histogram on the first frame where target is visible
        # If we have an initial bbox, we use it on frame 0 (or first frame read).
        histogram_initialized = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # --- Tracking Logic ---
            current_roi_image = image # Default to full image
            roi_offset_x = 0
            roi_offset_y = 0

            # If we have a target to track
            if last_bbox:
                # 1. Initialize Histogram if needed
                if not histogram_initialized:
                    # Crop to last_bbox
                    lx, ly, lw, lh = last_bbox
                    px, py = int(lx * w), int(ly * h)
                    pw, ph = int(lw * w), int(lh * h)

                    # Ensure within bounds
                    px = max(0, px); py = max(0, py)
                    pw = min(w - px, pw); ph = min(h - py, ph)

                    if pw > 0 and ph > 0:
                        roi = image[py:py+ph, px:px+pw]
                        pose_results = self.pose.process(roi)
                        if pose_results.pose_landmarks:
                            torso_roi = extract_torso_roi(roi, pose_results.pose_landmarks.landmark)
                            if torso_roi is not None:
                                target_hist = calculate_color_histogram(torso_roi)
                                histogram_initialized = True

                        # Fallback
                        if not histogram_initialized:
                            target_hist = calculate_color_histogram(roi)
                            histogram_initialized = True

                # 2. Track in current frame
                # Detect all candidates
                det_results = self.detector.detect(mp_image)
                best_bbox = None
                best_score = -1

                if det_results.detections and histogram_initialized:
                    for detection in det_results.detections:
                        # Check category
                        if detection.categories[0].category_name != 'person':
                            continue

                        bboxC = detection.bounding_box
                        # Convert to normalized
                        cx_norm, cy_norm = bboxC.origin_x / w, bboxC.origin_y / h
                        cw_norm, ch_norm = bboxC.width / w, bboxC.height / h

                        # Score 1: Spatial (IOU or Distance) - using Distance between centers
                        # Last center
                        lcx = last_bbox[0] + last_bbox[2]/2
                        lcy = last_bbox[1] + last_bbox[3]/2
                        # Current center
                        ccx = cx_norm + cw_norm/2
                        ccy = cy_norm + ch_norm/2

                        dist = np.sqrt((lcx - ccx)**2 + (lcy - ccy)**2)
                        score_dist = max(0, 1.0 - dist * 2) # Penalize distance heavily

                        # Score 2: Visual (Histogram)
                        # Extract ROI for candidate
                        px, py = int(cx_norm * w), int(cy_norm * h)
                        pw, ph = int(cw_norm * w), int(ch_norm * h)
                        px = max(0, px); py = max(0, py)
                        pw = min(w - px, pw); ph = min(h - py, ph)

                        score_hist = 0
                        if pw > 0 and ph > 0:
                            cand_roi = image[py:py+ph, px:px+pw]
                            cand_hist = calculate_color_histogram(cand_roi)
                            # Compare
                            score_hist = cv2.compareHist(target_hist, cand_hist, cv2.HISTCMP_CORREL)

                        # Combined Score
                        total_score = 0.6 * score_dist + 0.4 * score_hist

                        if total_score > best_score:
                            best_score = total_score
                            best_bbox = [cx_norm, cy_norm, cw_norm, ch_norm]

                # Update tracking state
                if best_bbox:
                    last_bbox = best_bbox
                    # Prepare ROI for Pose Analysis
                    lx, ly, lw, lh = last_bbox
                    px, py = int(lx * w), int(ly * h)
                    pw, ph = int(lw * w), int(lh * h)

                    # Add padding for Pose stability
                    pad_x = int(pw * 0.2)
                    pad_y = int(ph * 0.2)
                    px_pad = max(0, px - pad_x)
                    py_pad = max(0, py - pad_y)
                    pw_pad = min(w - px_pad, pw + 2 * pad_x)
                    ph_pad = min(h - py_pad, ph + 2 * pad_y)

                    if pw_pad > 0 and ph_pad > 0:
                        current_roi_image = image[py_pad:py_pad+ph_pad, px_pad:px_pad+pw_pad]
                        roi_offset_x = px_pad
                        roi_offset_y = py_pad
                else:
                    # Keep previous bbox as best guess, but maybe expand search next time?
                    pass

            # --- End Tracking Logic ---

            # Run Pose on the selected ROI
            results = self.pose.process(current_roi_image)

            frame_data = {
                "frame": frames_processed,
                "timestamp": frames_processed / fps if fps else 0,
                "landmarks": []
            }

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    # Adjust landmarks back to global coordinates if we cropped
                    if roi_offset_x > 0 or roi_offset_y > 0:
                        # lm.x is relative to crop width
                        global_x = (lm.x * current_roi_image.shape[1] + roi_offset_x) / w
                        global_y = (lm.y * current_roi_image.shape[0] + roi_offset_y) / h
                    else:
                        global_x = lm.x
                        global_y = lm.y

                    frame_data["landmarks"].append({
                        "x": global_x,
                        "y": global_y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })

            pose_data.append(frame_data)
            frames_processed += 1

            if frames_processed % 10 == 0 or frames_processed == frame_count:
                progress = (frames_processed / max(frame_count, 1)) * 0.7
                yield {
                    "status": "processing_video",
                    "progress": round(progress, 3),
                    "message": f"Processing video frame {frames_processed}/{frame_count}"
                }

        cap.release()
        pose_end = time.time()
        logger.info(f"Pose analysis took {pose_end - pose_start:.2f}s")

        # 2. Audio Analysis with Librosa
        yield {"status": "processing_audio", "progress": 0.7, "message": "Analyzing audio..."}
        audio_start = time.time()

        audio_data = {}
        try:
            # Librosa can be slow, especially loading
            y, sr = librosa.load(file_path)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            duration = librosa.get_duration(y=y, sr=sr)

            audio_data = {
                "tempo": float(tempo),
                "onset_env": onset_env.tolist()[::10], # Downsample for transmission
                "duration": duration,
                "sr": sr
            }
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            audio_data = {"error": str(e)}

        audio_end = time.time()
        logger.info(f"Audio analysis took {audio_end - audio_start:.2f}s")
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

        # Yield final result
        yield {"status": "complete", "progress": 1.0, "message": "Analysis complete", "result": final_result}

    def calculate_metrics(self, pose_data, audio_data):
        if not pose_data:
            return {}

        # 1. Koshi (MidHip Stability)
        # MidHip is roughly average of Left Hip (23) and Right Hip (24)
        mid_hip_y_variances = []
        for frame in pose_data:
            landmarks = frame.get("landmarks", [])
            if len(landmarks) > 24:
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                mid_hip_y = (left_hip['y'] + right_hip['y']) / 2
                mid_hip_y_variances.append(mid_hip_y)

        stability_score = 0.0
        if mid_hip_y_variances:
            # Lower variance means higher stability. Normalize.
            variance = np.var(mid_hip_y_variances)
            stability_score = max(0, 1.0 - (variance * 10)) # Heuristic scaling

        # 2. Kire (Jerk of Hands)
        # Calculate jerk (derivative of acceleration) for hands (15, 16)
        hand_velocities = []
        # Simplified: average movement of wrists per frame
        for i in range(1, len(pose_data)):
            curr = pose_data[i]["landmarks"]
            prev = pose_data[i-1]["landmarks"]
            if len(curr) > 16 and len(prev) > 16:
                # Left wrist (15) dist
                d_left = np.sqrt((curr[15]['x']-prev[15]['x'])**2 + (curr[15]['y']-prev[15]['y'])**2)
                # Right wrist (16) dist
                d_right = np.sqrt((curr[16]['x']-prev[16]['x'])**2 + (curr[16]['y']-prev[16]['y'])**2)
                hand_velocities.append((d_left + d_right) / 2)

        dynamism_score = 0.0
        if hand_velocities:
            # Kire implies ability to stop quickly (high deceleration) or move fast
            # We use max velocity / average velocity as a proxy for "dynamic range" of movement
            avg_vel = np.mean(hand_velocities)
            max_vel = np.max(hand_velocities)
            if avg_vel > 0:
                dynamism_score = min(1.0, (max_vel / avg_vel) / 5.0) # Heuristic

        # 3. Ma (Rhythm Harmony / Pause)
        # Check if stops in movement correlate with audio onsets
        rhythm_score = 0.5 # Default

        # Simple heuristic: "Ma" score based on ratio of low-movement frames (pauses)
        if hand_velocities:
            threshold = np.mean(hand_velocities) * 0.2
            pause_frames = sum(1 for v in hand_velocities if v < threshold)
            pause_ratio = pause_frames / len(hand_velocities)
            # Ideal "Ma" might be around 10-20% pause?
            rhythm_score = 1.0 - abs(pause_ratio - 0.15) * 2
            rhythm_score = max(0, min(1.0, rhythm_score))

        return {
            "stability_score": float(stability_score),
            "rhythm_score": float(rhythm_score),
            "dynamism_score": float(dynamism_score)
        }
