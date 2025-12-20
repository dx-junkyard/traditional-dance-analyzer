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

        if (currentFrameData && currentFrameData.landmarks) {
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
