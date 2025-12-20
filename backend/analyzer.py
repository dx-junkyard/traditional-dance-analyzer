import cv2
import mediapipe as mp
import numpy as np
import librosa
import json
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DanceAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    def analyze_video(self, file_path: str):
        """
        Generator function that yields progress updates and finally the result.
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

        pose_start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            frame_data = {
                "frame": frames_processed,
                "timestamp": frames_processed / fps if fps else 0,
                "landmarks": []
            }

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_data["landmarks"].append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })

            pose_data.append(frame_data)
            frames_processed += 1

            # Yield progress every 10 frames or if it's the last frame
            if frames_processed % 10 == 0 or frames_processed == frame_count:
                # Video processing is allocated 70% of the progress bar
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
