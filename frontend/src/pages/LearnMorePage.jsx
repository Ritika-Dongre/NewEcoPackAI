import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

const LearnMorePage = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-gradient-to-b from-white to-green-50 flex flex-col items-center justify-center px-6 py-16 relative overflow-hidden">
            {/* Background shapes */}
            <div className="absolute top-0 right-0 w-56 h-56 bg-green-200 rounded-full blur-3xl opacity-30 animate-pulse"></div>
            <div className="absolute bottom-0 left-0 w-72 h-72 bg-green-300 rounded-full blur-3xl opacity-20 animate-pulse"></div>

            <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1 }}
                className="max-w-5xl text-center md:text-left grid md:grid-cols-2 gap-12 items-center z-10"
            >
                {/* Text Section */}
                <div>
                    <h1 className="text-4xl md:text-5xl font-bold text-green-800 mb-6 text-center md:text-left">
                        Learn More About <span className="text-green-600">EcoPack AI</span>
                    </h1>
                    <p className="text-gray-700 mb-4 text-md md:text-lg leading-relaxed">
                        EcoPack AI revolutionizes sustainable packaging through the power of artificial intelligence. Our platform analyzes your product’s material composition, size, and sustainability goals to suggest the most eco-friendly packaging options available.
                    </p>
                    <p className="text-gray-700 mb-8 text-md md:text-lg leading-relaxed">
                        Whether you’re a small business looking to reduce costs or a large manufacturer aiming for carbon neutrality, EcoPack AI helps you make smarter, greener decisions at every stage of packaging.
                    </p>

                    <div className="flex flex-col md:flex-row gap-4 justify-center md:justify-start">
                        <button
                            onClick={() => navigate("/sustainability")}
                            className="bg-green-600 text-white px-8 py-3 rounded-lg font-semibold shadow-lg hover:bg-green-700 transform hover:scale-105 transition duration-300"
                        >
                            Explore Sustainability
                        </button>

                        <button
                            onClick={() => navigate(-1)}
                            className="bg-white text-green-600 border-2 border-green-600 px-8 py-3 rounded-lg font-semibold shadow-lg hover:bg-green-50 transform hover:scale-105 transition duration-300"
                        >
                            ← Back to Home
                        </button>
                    </div>
                </div>

                {/* Video Section  */}
                <motion.video
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="rounded-3xl shadow-2xl w-full max-w-sm mx-auto md:mx-0"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1.2 }}
                >
                    <source src="/videos/Eco-Friendly.mp4" type="video/mp4" />
                    Your browser does not support the video tag.
                </motion.video>
            </motion.div>
        </div>
    );
};

export default LearnMorePage;
