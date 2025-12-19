"use client";
import React, { useEffect, useRef, useState } from 'react';
import videojs from 'video.js';
import 'video.js/dist/video-js.css';
import Player from 'video.js/dist/types/player';

interface VideoAnalyzerProps {
    src?: string;
    poseData?: any[];
}

export const VideoAnalyzer: React.FC<VideoAnalyzerProps> = ({ src, poseData }) => {
    const videoRef = useRef<HTMLDivElement>(null);
    const playerRef = useRef<Player | null>(null);
    const [currentTime, setCurrentTime] = useState(0);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        if (!playerRef.current && videoRef.current) {
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
    }, [src, videoRef]);

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

    // Draw Pose Data Overlay
    useEffect(() => {
        if (!canvasRef.current || !poseData || !playerRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Find frame data closest to current time
        const currentFrameData = poseData.find((data) => Math.abs(data.timestamp - currentTime) < 0.1);

        if (currentFrameData && currentFrameData.landmarks) {
             // Draw connections (skeleton) - simplified example
            // In a real app, you'd map MediaPipe landmarks to canvas coordinates
            // Assuming landmarks are normalized 0-1

            const w = canvas.width;
            const h = canvas.height;

            ctx.fillStyle = 'red';
            ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
            ctx.lineWidth = 2;

            currentFrameData.landmarks.forEach((lm: any) => {
                const x = lm.x * w;
                const y = lm.y * h;

                if (lm.visibility > 0.5) {
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });

            // Draw skeleton lines (example: left arm)
            // You would implement full MediaPipe connectivity list here
        }

    }, [currentTime, poseData]);

    return (
        <div className="relative w-full max-w-4xl mx-auto">
            <div data-vjs-player>
                <div ref={videoRef} />
            </div>
            {/* Overlay Canvas */}
            <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
                width={1280}
                height={720}
            />
            <div className="mt-2 text-sm text-gray-600">
                Current Time: {currentTime.toFixed(2)}s
            </div>
        </div>
    );
};
