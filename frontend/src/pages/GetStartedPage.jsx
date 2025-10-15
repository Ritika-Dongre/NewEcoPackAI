import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

const GetStartedPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-white flex flex-col items-center justify-center px-6 py-16 relative overflow-hidden">
      {/* Background decorative shapes */}
      <div className="absolute top-10 left-10 w-48 h-48 bg-green-200 rounded-full blur-3xl opacity-30 animate-pulse"></div>
      <div className="absolute bottom-10 right-10 w-64 h-64 bg-green-300 rounded-full blur-3xl opacity-20 animate-pulse"></div>

      {/* Content */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
        className="text-center max-w-3xl z-10"
      >
        <h1 className="text-4xl md:text-5xl font-bold text-green-800 mb-6">
          Get Started with <span className="text-green-600">EcoPack AI</span>
        </h1>

        <p className="text-gray-700 mb-8 text-md md:text-lg leading-relaxed">
          Begin your sustainability journey with AI-powered insights. Upload your
          product details and discover how EcoPack AI helps you design
          eco-friendly packaging that reduces waste and optimizes resources —
          one package at a time.
        </p>

        {/* Embedded YouTube Video */}
        <motion.div
          className="relative w-full max-w-2xl mx-auto mb-10 rounded-3xl overflow-hidden shadow-2xl"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.2 }}
        >
          <div className="aspect-w-16 aspect-h-9">
            <iframe width="850" height="445" src="https://www.youtube.com/embed/42NTplnStQM" 
            title="The Evolution and Future of Sustainable Packaging | A Brief History" 
            frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
          </div>
        </motion.div>

        <div className="flex flex-col md:flex-row justify-center gap-4">
          <button
            onClick={() => alert("Redirecting to onboarding flow...")}
            className="bg-green-600 text-white px-8 py-3 rounded-lg font-semibold shadow-lg hover:bg-green-700 transform hover:scale-105 transition duration-300"
          >
            Start Your Journey
          </button>
          <button
            onClick={() => navigate(-1)}
            className="bg-white text-green-600 border-2 border-green-600 px-8 py-3 rounded-lg font-semibold shadow-lg hover:bg-green-50 transform hover:scale-105 transition duration-300"
          >
            ← Back to Home
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default GetStartedPage;
