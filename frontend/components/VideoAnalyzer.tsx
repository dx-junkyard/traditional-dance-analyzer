"use client";
import React, { useEffect, useRef, useState } from 'react';
import videojs from 'video.js';
import 'video.js/dist/video-js.css';
import Player from 'video.js/dist/types/player';

interface VideoAnalyzerProps {
    src?: string;
    poseData?: any[];
    // New props for selection mode
    selectionMode?: boolean;
    selectionImage?: string; // Base64
    candidates?: any[];
    onSelectCandidate?: (candidate: any) => void;
}

export const VideoAnalyzer: React.FC<VideoAnalyzerProps> = ({
    src,
    poseData,
    selectionMode,
    selectionImage,
    candidates,
    onSelectCandidate
}) => {
    const videoRef = useRef<HTMLDivElement>(null);
    const playerRef = useRef<Player | null>(null);
    const [currentTime, setCurrentTime] = useState(0);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Video Player Effect
    useEffect(() => {
        if (selectionMode) {
            // Destroy player if it exists to show static image
            if (playerRef.current) {
                playerRef.current.dispose();
                playerRef.current = null;
            }
            return;
        }

        if (!playerRef.current && videoRef.current && src) {
            const videoElement = document.createElement("video-js");
            videoElement.classList.add('vjs-big-play-centered');
            videoRef.current.appendChild(videoElement);

            const player = playerRef.current = videojs(videoElement, {
                controls: true,
                fluid: true,
                sources: src ? [{
                    src: src,
                    type: 'video/mp4'
                }] : []
            }, () => {
                videojs.log('player is ready');
                player.on('timeupdate', () => {
                    setCurrentTime(player.currentTime() || 0);
                });
            });
        } else {
            const player = playerRef.current;
            if (player && src) {
                player.src({ src: src, type: 'video/mp4' });
            }
        }
    }, [src, videoRef, selectionMode]);

    // Cleanup
    useEffect(() => {
        const player = playerRef.current;
        return () => {
            if (player && !player.isDisposed()) {
                player.dispose();
                playerRef.current = null;
            }
        };
    }, []);

    // Selection Mode Drawing (Static Image + Boxes)
    useEffect(() => {
        if (!selectionMode || !canvasRef.current || !selectionImage) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.src = `data:image/jpeg;base64,${selectionImage}`;
        img.onload = () => {
            // Draw image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Draw candidates
            if (candidates) {
                candidates.forEach((cand, idx) => {
                    // cand has x, y, width, height in normalized coords
                    const x = cand.x * canvas.width;
                    const y = cand.y * canvas.height;
                    const w = cand.width * canvas.width;
                    const h = cand.height * canvas.height;

                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, w, h);

                    // Label
                    ctx.fillStyle = '#00FF00';
                    ctx.font = '20px Arial';
                    ctx.fillText(`Person ${idx + 1}`, x, y - 5);
                });
            }
        };
    }, [selectionMode, selectionImage, candidates]);

    // Draw Pose Data Overlay (Analysis Mode)
    useEffect(() => {
        if (selectionMode || !canvasRef.current || !poseData || !playerRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Find frame data closest to current time
        const currentFrameData = poseData.find((data) => Math.abs(data.timestamp - currentTime) < 0.1);

        if (currentFrameData) {
            const w = canvas.width;
            const h = canvas.height;

            // MediaPipe Pose Connections (Partial based on mapped points)
            // Indices: 0:Nose, 11:LShoulder, 12:RShoulder, 13:LElbow, 14:RElbow,
            // 15:LWrist, 16:RWrist, 23:LHip, 24:RHip, 25:LKnee, 26:RKnee, 27:LAnkle, 28:RAnkle
            const connections = [
                [11, 12], // Shoulders
                [11, 13], [13, 15], // Left Arm
                [12, 14], [14, 16], // Right Arm
                [11, 23], [12, 24], // Torso sides
                [23, 24], // Hips
                [23, 25], [25, 27], // Left Leg
                [24, 26], [26, 28]  // Right Leg
            ];

            const drawSkeleton = (landmarks: any[], isTarget: boolean) => {
                const color = isTarget ? '#00FF00' : '#888888'; // Bright Green for Dancer, Gray for others
                ctx.strokeStyle = color;
                ctx.lineWidth = isTarget ? 4 : 2;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                // Draw Connections
                connections.forEach(([start, end]) => {
                    const l1 = landmarks[start];
                    const l2 = landmarks[end];

                    if (l1 && l2 && l1.visibility > 0.3 && l2.visibility > 0.3) {
                         ctx.beginPath();
                         ctx.moveTo(l1.x * w, l1.y * h);
                         ctx.lineTo(l2.x * w, l2.y * h);
                         ctx.stroke();
                    }
                });

                // Draw Label for Dancer
                if (isTarget) {
                    const head = landmarks[0]; // Nose
                    if (head && head.visibility > 0.3) {
                         ctx.fillStyle = '#00FF00';
                         ctx.font = 'bold 24px sans-serif';
                         ctx.textAlign = 'center';
                         ctx.fillText("Dancer", head.x * w, head.y * h - 20);
                    }
                }
            };

            // Handle new 'people' structure or fallback to 'landmarks'
            if (currentFrameData.people && currentFrameData.people.length > 0) {
                currentFrameData.people.forEach((person: any) => {
                    drawSkeleton(person.landmarks, person.is_target);
                });
            } else if (currentFrameData.landmarks) {
                // Legacy/Single target fallback
                drawSkeleton(currentFrameData.landmarks, true);
            }
        }
    }, [currentTime, poseData, selectionMode]);

    const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!selectionMode || !onSelectCandidate || !candidates) return;

        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;

        const scaleX = canvasRef.current!.width / rect.width;
        const scaleY = canvasRef.current!.height / rect.height;

        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        // Find clicked candidate
        const w = canvasRef.current!.width;
        const h = canvasRef.current!.height;

        // Check in reverse order (topmost first if overlapping)
        for (let i = candidates.length - 1; i >= 0; i--) {
            const cand = candidates[i];
            const cx = cand.x * w;
            const cy = cand.y * h;
            const cw = cand.width * w;
            const ch = cand.height * h;

            if (clickX >= cx && clickX <= cx + cw && clickY >= cy && clickY <= cy + ch) {
                onSelectCandidate(cand);
                break;
            }
        }
    };

    return (
        <div className="relative w-full max-w-4xl mx-auto aspect-video bg-black">
             {/* Player Container - Only show if not selection mode or we need to hide it */}
            <div data-vjs-player style={{ display: selectionMode ? 'none' : 'block' }}>
                <div ref={videoRef} />
            </div>

            {/* Overlay Canvas */}
            <canvas
                ref={canvasRef}
                className={`absolute top-0 left-0 w-full h-full ${selectionMode ? 'cursor-pointer' : 'pointer-events-none'}`}
                width={1280}
                height={720}
                onClick={handleCanvasClick}
            />
            {!selectionMode && (
                <div className="mt-2 text-sm text-gray-600">
                    Current Time: {currentTime.toFixed(2)}s
                </div>
            )}
             {selectionMode && (
                <div className="absolute bottom-4 left-0 w-full text-center pointer-events-none">
                    <span className="bg-black/70 text-white px-4 py-2 rounded-full text-lg font-bold">
                        Click on the dancer to track
                    </span>
                </div>
            )}
        </div>
    );
};
