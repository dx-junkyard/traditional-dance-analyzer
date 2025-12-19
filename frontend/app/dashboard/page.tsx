"use client";
import React, { useState, useEffect } from 'react';
import { VideoAnalyzer } from '@/components/VideoAnalyzer';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Upload, Activity, Music, Zap } from 'lucide-react';
import axios from 'axios';

export default function DashboardPage() {
    const [file, setFile] = useState<File | null>(null);
    const [analysisResult, setAnalysisResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);

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
        } finally {
            setLoading(false);
        }
    };

    // Prepare chart data from pose_data (mocking a time series metric)
    const chartData = analysisResult?.pose_data?.map((frame: any) => ({
        timestamp: frame.timestamp,
        stability: Math.random() * 0.5 + 0.5, // Mock dynamic data
        energy: Math.random() * 100
    })) || [];

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <header className="mb-8 flex justify-between items-center">
                <h1 className="text-3xl font-bold text-gray-800">Traditional Dance Analyzer</h1>
                <div className="flex items-center gap-4">
                    <span className="text-sm text-gray-600">User: Dance Master</span>
                    <button className="bg-gray-200 px-4 py-2 rounded text-sm hover:bg-gray-300">Logout</button>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Video & Upload */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                        <div className="mb-4">
                            <VideoAnalyzer src={videoUrl || undefined} poseData={analysisResult?.pose_data} />
                        </div>

                        <div className="flex gap-4 items-center">
                             <input
                                type="file"
                                accept="video/*"
                                onChange={handleFileChange}
                                className="block w-full text-sm text-gray-500
                                file:mr-4 file:py-2 file:px-4
                                file:rounded-full file:border-0
                                file:text-sm file:font-semibold
                                file:bg-violet-50 file:text-violet-700
                                hover:file:bg-violet-100"
                            />
                            <button
                                onClick={handleUpload}
                                disabled={!file || loading}
                                className="flex items-center gap-2 bg-blue-600 text-white px-6 py-2 rounded-full hover:bg-blue-700 disabled:opacity-50"
                            >
                                <Upload size={18} />
                                {loading ? `Analyzing ${progress}%` : 'Analyze'}
                            </button>
                        </div>
                        {loading && (
                            <div className="w-full bg-gray-200 rounded-full h-2.5 mt-4">
                                <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
                            </div>
                        )}
                    </div>

                     {/* Time Series Analysis */}
                    {analysisResult && (
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Activity size={20} className="text-blue-500"/> Movement Dynamics
                            </h3>
                            <div className="h-64 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="timestamp" label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5 }} />
                                        <YAxis />
                                        <Tooltip />
                                        <Legend />
                                        <Line type="monotone" dataKey="stability" stroke="#8884d8" name="Koshi Stability" />
                                        <Line type="monotone" dataKey="energy" stroke="#82ca9d" name="Energy Flow" />
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
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                                <h3 className="text-lg font-semibold mb-4">Core Metrics</h3>
                                <div className="space-y-6">
                                    <div>
                                        <div className="flex justify-between mb-1">
                                            <span className="text-sm font-medium text-gray-700">Koshi Stability</span>
                                            <span className="text-sm font-medium text-blue-600">{(analysisResult.metrics.stability_score * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-2">
                                            <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${analysisResult.metrics.stability_score * 100}%` }}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between mb-1">
                                            <span className="text-sm font-medium text-gray-700">Rhythm Harmony</span>
                                            <span className="text-sm font-medium text-purple-600">{(analysisResult.metrics.rhythm_score * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-2">
                                            <div className="bg-purple-600 h-2 rounded-full" style={{ width: `${analysisResult.metrics.rhythm_score * 100}%` }}></div>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between mb-1">
                                            <span className="text-sm font-medium text-gray-700">Jo-Ha-Kyu (Dynamism)</span>
                                            <span className="text-sm font-medium text-orange-600">{(analysisResult.metrics.dynamism_score * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-2">
                                            <div className="bg-orange-600 h-2 rounded-full" style={{ width: `${analysisResult.metrics.dynamism_score * 100}%` }}></div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                                <h3 className="text-lg font-semibold mb-2">AI Feedback</h3>
                                <p className="text-gray-600 text-sm leading-relaxed">
                                    Your "Ma" (pause) at 0:15 was excellent. However, the center of gravity tends to float upwards during the "Kyu" phase (fast movement). Focus on grounding your hips.
                                </p>
                            </div>
                        </>
                    ) : (
                         <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col items-center justify-center h-64 text-center">
                            <Zap className="text-gray-300 w-12 h-12 mb-4" />
                            <p className="text-gray-500">Upload a video to see analysis insights.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
