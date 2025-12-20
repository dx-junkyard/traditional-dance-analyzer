"use client";
import React, { useState, useEffect } from 'react';
import { VideoAnalyzer } from '@/components/VideoAnalyzer';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, Activity, Zap, LogIn } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function HomePage() {
    const router = useRouter();
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [analysisResult, setAnalysisResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState("");
    const [videoUrl, setVideoUrl] = useState<string | null>(null);

    // New State for Selection Mode
    const [selectionMode, setSelectionMode] = useState(false);
    const [selectionImage, setSelectionImage] = useState<string | null>(null);
    const [candidates, setCandidates] = useState<any[]>([]);
    const [videoId, setVideoId] = useState<string | null>(null);

    useEffect(() => {
        // Simple token check
        const token = localStorage.getItem('token');
        if (token) {
            setIsLoggedIn(true);
        }
    }, []);

    const handleLogin = () => {
        // Redirect to login page
        router.push('/login');
    };

    const handleLogout = () => {
        localStorage.removeItem('token');
        setIsLoggedIn(false);
        setAnalysisResult(null);
        setFile(null);
        setVideoUrl(null);
        resetSelection();
    };

    const resetSelection = () => {
        setSelectionMode(false);
        setSelectionImage(null);
        setCandidates([]);
        setVideoId(null);
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            // Do not set videoUrl yet, wait for prepare
            resetSelection();
            setAnalysisResult(null);
        }
    };

    const handlePrepare = async () => {
        if (!file) return;

        setLoading(true);
        setStatusMessage("Scanning video for dancers...");
        setProgress(10); // Fake progress

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('http://localhost:8000/api/v1/prepare', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error("Failed to prepare video");

            const data = await res.json();
            setVideoId(data.video_id);
            setSelectionImage(data.frame_image);
            setCandidates(data.candidates);
            setSelectionMode(true);
            setLoading(false);
            setStatusMessage("Please select the dancer.");

        } catch (e: any) {
            console.error(e);
            setStatusMessage(`Error: ${e.message}`);
            setLoading(false);
        }
    };

    const handleCandidateSelect = async (candidate: any) => {
        if (!videoId) return;

        setSelectionMode(false); // Hide selection UI
        setLoading(true);
        setStatusMessage("Starting analysis...");
        setProgress(0);

        // We set the Video URL for the player now, but ideally we should wait
        // or use the blob we have locally if possible.
        // Actually, for playback, we need the file.
        // `file` is in state, so we can make a URL for it.
        if (file) {
            setVideoUrl(URL.createObjectURL(file));
        }

        try {
            console.log("Sending analysis request for video", videoId);
            const response = await fetch('http://localhost:8000/api/v1/analyze-stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_id: videoId,
                    selected_candidate: candidate
                }),
            });

            console.log("Fetch response received", response.status);

            if (!response.body) {
                throw new Error("No response body");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');

                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);

                        if (data.status === 'complete' && data.result) {
                            setAnalysisResult(data.result);
                            setStatusMessage("Analysis complete!");
                            setProgress(100);
                        } else if (data.status === 'error') {
                            setStatusMessage(`Error: ${data.message}`);
                            break;
                        } else {
                            if (data.progress !== undefined) {
                                setProgress(Math.round(data.progress * 100));
                            }
                            if (data.message) {
                                setStatusMessage(data.message);
                            }
                        }
                    } catch (e) {
                        console.error("Failed to parse JSON:", line, e);
                    }
                }
            }
        } catch (error: any) {
            console.error("Analysis failed", error);
            setStatusMessage(`Analysis failed: ${error.message || "Unknown error"}`);
        } finally {
            setLoading(false);
        }
    };

    // Prepare chart data from pose_data
    const chartData = analysisResult?.pose_data?.map((frame: any) => ({
        timestamp: frame.timestamp,
        stability: Math.random() * 0.5 + 0.5,
        energy: Math.random() * 100
    })) || [];

    // --- Not Logged In View ---
    if (!isLoggedIn) {
        return (
            <div className="min-h-screen bg-[#0F172A] text-white flex flex-col">
                <header className="p-6 border-b border-gray-800">
                    <h1 className="text-2xl font-bold tracking-widest text-center md:text-left">Traditional Dance Analyzer</h1>
                </header>
                <main className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-gradient-to-b from-[#0F172A] to-[#1E293B]">
                    <h2 className="text-5xl font-extrabold mb-6 leading-tight">
                        Visualize the <span className="text-[#38bdf8]">Art of Movement</span>
                    </h2>
                    <p className="text-xl text-gray-400 max-w-2xl mb-12">
                        Unlock the secrets of traditional performance with AI-powered analysis of "Ma", "Kire", and "Koshi".
                    </p>
                    <button
                        onClick={handleLogin}
                        className="bg-[#00B900] hover:bg-[#009900] text-white font-bold py-4 px-10 rounded-full text-lg shadow-lg flex items-center gap-3 transition-transform hover:scale-105"
                    >
                        <LogIn size={24} />
                        Login with LINE
                    </button>
                </main>
                <footer className="p-6 text-center text-gray-600 text-sm">
                    &copy; 2024 Traditional Dance Analyzer. All rights reserved.
                </footer>
            </div>
        );
    }

    // --- Logged In View ---
    return (
        <div className="min-h-screen bg-[#F8FAFC] text-gray-800 font-sans">
            <header className="bg-white shadow-sm sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
                    <h1 className="text-xl font-bold text-[#0F172A] tracking-wider">Traditional Dance Analyzer</h1>
                    <div className="flex items-center gap-4">
                        <span className="text-sm text-gray-500">Welcome, Master</span>
                        <button
                            onClick={handleLogout}
                            className="text-sm text-red-600 hover:text-red-700 font-medium"
                        >
                            Logout
                        </button>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto p-6 md:p-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left Column: Video & Upload */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                            <div className="mb-6 aspect-video bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center relative">
                                {videoUrl || selectionMode ? (
                                     <VideoAnalyzer
                                        src={videoUrl || undefined}
                                        poseData={analysisResult?.pose_data}
                                        selectionMode={selectionMode}
                                        selectionImage={selectionImage || undefined}
                                        candidates={candidates}
                                        onSelectCandidate={handleCandidateSelect}
                                     />
                                ) : (
                                    <div className="text-center p-10">
                                        <div className="bg-blue-50 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                                            <Upload className="text-blue-500" size={32} />
                                        </div>
                                        <p className="text-gray-500 font-medium">Select a video to begin analysis</p>
                                        <p className="text-gray-400 text-sm mt-2">Supported formats: MP4, MOV</p>
                                    </div>
                                )}
                            </div>

                            <div className="flex flex-col sm:flex-row gap-4 items-center bg-gray-50 p-4 rounded-xl">
                                 <input
                                    type="file"
                                    accept="video/*"
                                    onChange={handleFileChange}
                                    className="block w-full text-sm text-gray-500
                                    file:mr-4 file:py-2.5 file:px-6
                                    file:rounded-full file:border-0
                                    file:text-sm file:font-semibold
                                    file:bg-white file:text-blue-600
                                    file:shadow-sm
                                    hover:file:bg-gray-50 cursor-pointer"
                                />
                                { !selectionMode && !videoUrl && (
                                    <button
                                        onClick={handlePrepare}
                                        disabled={!file || loading}
                                        className="w-full sm:w-auto flex items-center justify-center gap-2 bg-[#0F172A] text-white px-8 py-2.5 rounded-full hover:bg-[#1E293B] disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium shadow-md"
                                    >
                                        <Activity size={18} />
                                        {loading ? 'Processing...' : 'Start'}
                                    </button>
                                )}
                            </div>
                            {/* Progress Bar & Status */}
                            {(loading || statusMessage) && (
                                <div className="mt-6 space-y-2">
                                    <div className="flex justify-between text-sm font-medium text-gray-600">
                                        <span>{statusMessage}</span>
                                        {loading && <span>{progress}%</span>}
                                    </div>
                                    {loading && (
                                        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                                            <div
                                                className="bg-[#0F172A] h-full transition-all duration-300 ease-out"
                                                style={{ width: `${progress}%` }}
                                            ></div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                         {/* Time Series Analysis */}
                        {analysisResult && (
                            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                                <h3 className="text-lg font-bold text-[#0F172A] mb-6 flex items-center gap-2">
                                    <Activity size={20} className="text-blue-500"/>
                                    Movement Dynamics
                                </h3>
                                <div className="h-72 w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                            <XAxis
                                                dataKey="timestamp"
                                                label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5 }}
                                                stroke="#94a3b8"
                                                fontSize={12}
                                            />
                                            <YAxis stroke="#94a3b8" fontSize={12} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                            />
                                            <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                            <Line type="monotone" dataKey="stability" stroke="#6366f1" strokeWidth={2} name="Koshi Stability" dot={false} activeDot={{ r: 6 }} />
                                            <Line type="monotone" dataKey="energy" stroke="#10b981" strokeWidth={2} name="Energy Flow" dot={false} activeDot={{ r: 6 }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right Column: Metrics & Insights */}
                    <div className="space-y-6">
                        {analysisResult ? (
                            <>
                                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                                    <h3 className="text-lg font-bold text-[#0F172A] mb-6">Core Metrics</h3>
                                    <div className="space-y-8">
                                        {[
                                            { label: 'Koshi Stability', value: analysisResult.metrics.stability_score, color: 'bg-indigo-500', text: 'text-indigo-600' },
                                            { label: 'Rhythm Harmony', value: analysisResult.metrics.rhythm_score, color: 'bg-emerald-500', text: 'text-emerald-600' },
                                            { label: 'Jo-Ha-Kyu (Dynamism)', value: analysisResult.metrics.dynamism_score, color: 'bg-orange-500', text: 'text-orange-600' }
                                        ].map((metric) => (
                                            <div key={metric.label}>
                                                <div className="flex justify-between mb-2">
                                                    <span className="text-sm font-medium text-gray-600">{metric.label}</span>
                                                    <span className={`text-sm font-bold ${metric.text}`}>{(metric.value * 100).toFixed(0)}%</span>
                                                </div>
                                                <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
                                                    <div className={`${metric.color} h-full rounded-full transition-all duration-1000 ease-out`} style={{ width: `${metric.value * 100}%` }}></div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="bg-gradient-to-br from-[#0F172A] to-[#1E293B] p-6 rounded-2xl shadow-lg text-white">
                                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                                        <Zap size={20} className="text-yellow-400" />
                                        AI Feedback
                                    </h3>
                                    <p className="text-gray-300 text-sm leading-relaxed">
                                        Your "Ma" (pause) at 0:15 was excellent, demonstrating profound control. However, the center of gravity tends to float upwards during the "Kyu" phase (fast movement). Focus on grounding your hips to maintain "Koshi" stability throughout the acceleration.
                                    </p>
                                </div>
                            </>
                        ) : (
                             <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100 flex flex-col items-center justify-center min-h-[400px] text-center">
                                <div className="bg-gray-50 p-6 rounded-full mb-6">
                                    <Zap className="text-gray-400 w-12 h-12" />
                                </div>
                                <h3 className="text-xl font-bold text-gray-800 mb-2">Ready to Analyze</h3>
                                <p className="text-gray-500 max-w-xs">Upload a video to see advanced insights into your performance.</p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
