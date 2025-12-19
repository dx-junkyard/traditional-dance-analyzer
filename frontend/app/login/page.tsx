"use client";
import React, { useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import axios from 'axios';

function LoginContent() {
    const router = useRouter();
    const searchParams = useSearchParams();

    const handleLineLogin = () => {
        // Redirect to LINE Login Authorization URL
        // In real scenario: https://access.line.me/oauth2/v2.1/authorize...
        // Here we mock it by redirecting to self with a mock code
        router.push('/login?code=mock_line_auth_code_12345');
    };

    useEffect(() => {
        const code = searchParams.get('code');
        if (code) {
            // Exchange code for token via backend
            axios.post('http://localhost:8000/api/v1/users/login', null, {
                params: { code }
            }).then(response => {
                console.log("Login success:", response.data);
                // Save token (localStorage/cookie) - Mocking
                localStorage.setItem('token', response.data.token);
                router.push('/dashboard');
            }).catch(err => {
                console.error("Login failed:", err);
            });
        }
    }, [searchParams, router]);

    return (
        <div className="bg-white p-8 rounded-lg shadow-md w-96 text-center">
            <h1 className="text-2xl font-bold mb-6 text-gray-800">Welcome to Dance Analyzer</h1>
            <p className="mb-6 text-gray-600">Please sign in to continue</p>

            <button
                onClick={handleLineLogin}
                className="w-full bg-[#00B900] text-white font-bold py-3 px-4 rounded hover:bg-[#009900] transition duration-200 flex items-center justify-center gap-2"
            >
                Login with LINE
            </button>
        </div>
    );
}

export default function LoginPage() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100">
            <Suspense fallback={<div>Loading...</div>}>
                <LoginContent />
            </Suspense>
        </div>
    );
}
