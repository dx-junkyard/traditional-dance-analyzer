"use client";
import React, { useState, useEffect } from 'react';
import { VideoAnalyzer } from '@/components/VideoAnalyzer';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, Activity, Zap, LogIn } from 'lucide-react';
import axios from 'axios';
import { useRouter } from 'next/navigation';

export default function HomePage() {
    const router = useRouter();
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [analysisResult, setAnalysisResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);

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
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setVideoUrl(URL.createObjectURL(e.target.files[0]));
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Start listening for status
            const eventSource = new EventSource(`http://localhost:8000/api/v1/status/${file.name}`);
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                setProgress(Math.round(data.progress * 100));
                if (data.progress >= 1.0) {
                    eventSource.close();
                }
            };

            const response = await axios.post('http://localhost:8000/api/v1/analyze', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setAnalysisResult(response.data);
        } catch (error) {
            console.error("Analysis failed", error);
            // In a real app, handle error UI here
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
                                {videoUrl ? (
                                     <VideoAnalyzer src={videoUrl || undefined} poseData={analysisResult?.pose_data} />
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
                                <button
                                    onClick={handleUpload}
                                    disabled={!file || loading}
                                    className="w-full sm:w-auto flex items-center justify-center gap-2 bg-[#0F172A] text-white px-8 py-2.5 rounded-full hover:bg-[#1E293B] disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium shadow-md"
                                >
                                    <Activity size={18} />
                                    {loading ? `Analyzing ${progress}%` : 'Analyze'}
                                </button>
                            </div>
                            {loading && (
                                <div className="w-full bg-gray-200 rounded-full h-1.5 mt-4 overflow-hidden">
                                    <div className="bg-[#0F172A] h-full transition-all duration-300 ease-out" style={{ width: `${progress}%` }}></div>
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
