import cv2
import mediapipe as mp
import numpy as np
import librosa
import json
import tempfile
import os
from fastapi import UploadFile

class DanceAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    async def analyze_video(self, file: UploadFile, status_callback=None):
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 1. Pose Analysis with MediaPipe
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pose_data = []
        frames_processed = 0

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

            if status_callback and frames_processed % 30 == 0:
                await status_callback(frames_processed / frame_count * 0.5) # 50% for visual

        cap.release()

        # 2. Audio Analysis with Librosa
        try:
            y, sr = librosa.load(temp_path)
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
            print(f"Audio analysis failed: {e}")
            audio_data = {"error": str(e)}

        if status_callback:
            await status_callback(1.0)

        # Cleanup
        os.remove(temp_path)
        os.rmdir(temp_dir)

        metrics = self.calculate_metrics(pose_data, audio_data)

        return {
            "pose_data": pose_data,
            "audio_data": audio_data,
            "metrics": metrics
        }

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
