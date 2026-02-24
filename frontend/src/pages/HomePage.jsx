import React, { useState } from "react";
import { motion } from "framer-motion";
import { HiMenu, HiX } from "react-icons/hi";
import { useNavigate } from "react-router-dom";

const HomePage = () => {
    const [menuOpen, setMenuOpen] = useState(false);
    const navigate = useNavigate(); // hook for navigation

    const scrollToSection = (id) => {
        const element = document.getElementById(id);
        if (element) {
            element.scrollIntoView({ behavior: "smooth" });
            setMenuOpen(false);
        }
    };

    const videoSource = "/videos/EcoPackaging.mov";

    return (
        <div className="w-full min-h-screen bg-gradient-to-b from-green-50 to-white font-sans overflow-x-hidden-hidden">

            {/* Navbar (Unchanged) */}
            <nav className="w-full bg-green-50 shadow-md fixed top-0 left-0 z-50">
                <div className="max-w-7xl mx-auto px-8 md:px-20 flex items-center justify-between h-20">
                    <div className="flex items-center space-x-3 cursor-pointer">
                        <img
                            src="/images/ecoLogo.png"
                            alt="EcoPack AI Logo"
                            className="w-12 h-12 rounded-full object-cover shadow-md hover:scale-110 transform transition duration-300"
                        />
                        <span className="text-2xl font-bold text-green-700">
                            EcoPack <span className="text-green-600">AI</span>
                        </span>
                    </div>

                    {/* Desktop Menu */}
                    <ul className="hidden md:flex space-x-8 text-green-700 font-semibold">
                        <li className="hover:text-green-900 cursor-pointer transition" onClick={() => scrollToSection("hero")}>Home</li>
                        <li className="hover:text-green-900 cursor-pointer transition" onClick={() => scrollToSection("about")}>About</li>
                        {/* <li className="hover:text-green-900 cursor-pointer transition" onClick={() => navigate("/login")}>Login/SignUp</li> */}
                        <li className="hover:text-green-900 cursor-pointer transition" onClick={() => navigate("/upload")}>Upload</li>

                    </ul>

                    {/* Mobile Menu Button */}
                    <div className="md:hidden">
                        <button onClick={() => setMenuOpen(!menuOpen)}>
                            {menuOpen ? <HiX size={28} /> : <HiMenu size={28} />}
                        </button>
                    </div>
                </div>

                {menuOpen && (
                    <div className="md:hidden bg-green-50 shadow-md px-4 py-4 flex flex-col space-y-4 text-green-700 font-semibold">
                        <li onClick={() => scrollToSection("hero")} className="cursor-pointer hover:text-green-900">Home</li>
                        <li onClick={() => scrollToSection("about")} className="cursor-pointer hover:text-green-900">About</li>
                        {/* <li className="cursor-pointer hover:text-green-900">Login/Signup</li> */}
                    </div>
                )}
            </nav>
            <section id="hero" className="flex flex-col md:flex-row items-center justify-between px-8 md:px-20 h-screen pt-20 relative overflow-hidden">
                
                {/* 1. VIDEO ELEMENT - Full Background */}
                <video 
                    autoPlay 
                    loop 
                    muted 
                    playsInline 
                    className="absolute inset-0 w-full h-full object-cover z-0 opacity-20" 
                    aria-label="Background animation of eco-friendly packaging"
                >
                    <source src={videoSource} type="video/mp4" /> 
                    Your browser does not support the video tag.
                </video>
                <motion.div className="md:w-1/2 space-y-6 z-10 text-center md:text-left" initial={{ opacity: 0, x: -50 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 1 }}>
                    <h1 className="text-3xl md:text-5xl font-bold text-green-900 leading-tight">
                        Welcome to <span className="text-green-600">EcoPack AI</span>
                    </h1>
                    <p className="text-gray-700 text-md md:text-lg">
                        Revolutionizing packaging with AI-driven eco-friendly solutions. Reduce waste, save resources, and make a positive impact on our planet.
                    </p>
                    <div className="flex flex-col md:flex-row space-y-3 md:space-y-0 md:space-x-4 justify-center md:justify-start">
                        <button
                            onClick={() => navigate("/get-started")}
                            className="bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg transform hover:scale-105 hover:bg-green-700 transition duration-300"
                        >
                            Get Started
                        </button>
                        <button
                            onClick={() => navigate("/learn-more")}
                            className="bg-white text-green-600 border-2 border-green-600 px-6 py-3 rounded-lg shadow-lg transform hover:scale-105 hover:bg-green-50 transition duration-300"
                        >
                            Learn More
                        </button>
                    </div>
                </motion.div>

                <motion.div className="md:w-1/2 mt-10 md:mt-0 z-10 flex justify-center" initial={{ opacity: 0, x: 50 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 1 }}>
                    <img 
                      src="/images/image.png"
                      alt="Eco Packaging Mockup" 
                      className="rounded-3xl shadow-2xl hover:scale-105 transform transition duration-500 w-full max-w-md hidden md:block" // Optional: Keep a mockup image
                    />
                </motion.div>
                <div className="absolute top-0 left-0 w-40 h-40 bg-green-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse"></div>
                <div className="absolute bottom-0 right-0 w-56 h-56 bg-green-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
            </section>

            {/* Features Section  */}
            <section className="bg-green-50 py-20 px-8 md:px-20 text-center">
                <h2 className="text-2xl md:text-4xl font-bold text-green-800 mb-12">
                    Features
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {[
                        { title: "AI-Powered Analysis", description: "Smart recommendations for sustainable packaging based on AI insights." },
                        { title: "Eco-Friendly Materials", description: "Discover biodegradable, recyclable, and reusable packaging options." },
                        { title: "Track & Reduce Waste", description: "Monitor packaging impact and optimize for minimal environmental footprint." },
                    ].map((feature, idx) => (
                        <motion.div key={idx} className="bg-white p-6 md:p-8 rounded-3xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition duration-500" initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.8, delay: idx * 0.2 }}>
                            <h2 className="text-xl md:text-2xl font-semibold text-green-700 mb-2 md:mb-4">{feature.title}</h2>
                            <p className="text-gray-600 text-sm md:text-md">{feature.description}</p>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* How It Works Section */}
            <section className="bg-green-50 py-20 px-8 md:px-20 text-center">
                <h2 className="text-2xl md:text-4xl font-bold text-green-800 mb-12">
                    How EcoPack AI Works
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {[
                        { step: "01", title: "Upload Product Info", description: "Provide details about your product and packaging requirements." },
                        { step: "02", title: "AI Suggests Packaging", description: "Our AI analyzes and recommends the most eco-friendly packaging solutions." },
                        { step: "03", title: "Track Impact", description: "Monitor environmental impact and optimize packaging continuously." },
                    ].map((step, idx) => (
                        <motion.div key={idx} className="bg-white p-6 md:p-8 rounded-3xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-500" initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.8, delay: idx * 0.2 }}>
                            <div className="text-green-600 text-3xl md:text-4xl font-bold mb-4">{step.step}</div>
                            <h3 className="text-xl md:text-2xl font-semibold mb-2">{step.title}</h3>
                            <p className="text-gray-600 text-sm md:text-md">{step.description}</p>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* About Section */}
            <section id="about" className="px-6 md:px-20 py-20 bg-green-100 relative">
                <h2 className="text-3xl md:text-4xl font-bold text-green-800 text-center mb-12">About EcoPack AI</h2>
                <div className="flex flex-col md:flex-row items-center gap-8 md:gap-16">

                    {/* Text */}
                    <motion.div className="md:w-1/2 space-y-6 text-center md:text-left" initial={{ opacity: 0, x: -50 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 1 }}>
                        <h3 className="text-2xl md:text-3xl font-semibold text-green-700">Revolutionizing Sustainable Packaging</h3>
                        <p className="text-gray-700 text-sm md:text-md">
                            EcoPack AI is on a mission to transform packaging for businesses of all sizes. Using AI technology, we provide sustainable packaging solutions that are intelligent, eco-friendly, and cost-effective.
                        </p>
                        <p className="text-gray-700 text-sm md:text-md">
                            <strong>Why Choose EcoPack AI?</strong> We combine data-driven AI analysis with biodegradable, recyclable, and reusable materials. This helps businesses minimize environmental impact while maximizing efficiency.
                        </p>
                        <div className="text-gray-700 text-sm md:text-md">
                            <strong>Key Benefits:</strong>
                            <ul className="list-disc ml-6 mt-2 space-y-1 text-gray-600">
                                <li>Reduce plastic and packaging waste.</li>
                                <li>Optimize costs with AI insights.</li>
                                <li>Boost brand reputation with sustainable practices.</li>
                                <li>Track and measure environmental impact.</li>
                            </ul>
                        </div>
                        <p className="text-gray-700 text-sm md:text-md">
                            With EcoPack AI, packaging smarter isn’t just an option — it’s a responsibility to the planet and your customers.
                        </p>
                        <button className="bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg transform hover:scale-105 hover:bg-green-700 transition duration-300 mt-3">
                            Learn More
                        </button>
                    </motion.div>

                    {/* Image */}
                    <motion.div className="md:w-1/2 flex justify-center" initial={{ opacity: 0, x: 50 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 1 }}>
                        <img src="https://www.ecopack.bg/web/files/richeditor/Ecopack_diagram_ENG_preview_V05.jpg" alt="EcoPack AI Mission" className="rounded-3xl shadow-2xl hover:scale-105 transform transition duration-500 w-full max-w-sm" />
                    </motion.div>
                </div>
            </section>

            {/* Call to Action  */}
            <motion.section className="bg-green-600 text-white py-20 text-center relative overflow-hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
                <h2 className="text-2xl md:text-4xl font-bold mb-4">Ready to Make a Difference?</h2>
                <p className="text-sm md:text-lg mb-6">Join us in creating smarter, sustainable packaging solutions.</p>
                <button className="bg-white text-green-600 px-4 md:px-8 py-1 md:py-2 rounded-lg font-semibold shadow-lg transform hover:scale-105 hover:bg-green-50 transition duration-300">
                    Try EcoPack AI Now
                </button>
            </motion.section>

            {/* Footer (Unchanged) */}
            <footer className="bg-green-800 text-white py-6 px-8 md:px-20 text-center">
                <p>© 2025 EcoPack AI. All rights reserved.</p>
            </footer>

        </div>
    );
};

export default HomePage;
