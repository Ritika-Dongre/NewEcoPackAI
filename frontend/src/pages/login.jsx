// ...existing code...
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';


export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();

    async function handleSubmit(e) {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data?.message || 'Login failed');
            // store token (adjust to your auth flow)
            localStorage.setItem('token', data.token);
            navigate('/');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="w-full max-w-md bg-white p-8 rounded-lg shadow">
                <h1 className="text-2xl font-semibold mb-6 text-gray-800">Sign in</h1>
                {error && <div className="mb-4 text-sm text-red-600">{error}</div>}
                <form onSubmit={handleSubmit} className="space-y-4">
                    <label className="block">
                        <span className="text-sm text-gray-700">Email</span>
                        <input
                            type="email"
                            required
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="mt-1 block w-full rounded border-gray-200 shadow-sm focus:ring-2 focus:ring-green-400"
                        />
                    </label>

                    <label className="block">
                        <span className="text-sm text-gray-700">Password</span>
                        <input
                            type="password"
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="mt-1 block w-full rounded border-gray-200 shadow-sm focus:ring-2 focus:ring-green-400"
                        />
                    </label>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-2 px-4 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-60"
                    >
                        {loading ? 'Signing in...' : 'Sign in'}
                    </button>
                </form>

                <p className="mt-6 text-sm text-center text-gray-600">
                    Don't have an account?{' '}
                    <Link to="/signup" className="text-green-600 hover:underline">
                        Create one
                    </Link>
                </p>
            </div>
        </div>
    );
}