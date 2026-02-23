// App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx"; 
import UploadPage from "./pages/UploadPage.jsx";
import GetStartedPage from "./pages/GetStartedPage";
import LearnMorePage from "./pages/LearnMorePage";
import SustainabilityPage from "./pages/SustainabilityPage";
import SignupPage from "./pages/signup.jsx";
import LoginPage from "./pages/login.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import VendorsPage from "./pages/VendorsPage.jsx";


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />      
        <Route path="/upload" element={<UploadPage />} /> 
        <Route path="/get-started" element={<GetStartedPage />} />
        <Route path="/learn-more" element={<LearnMorePage />} />
        <Route path="/sustainability" element={<SustainabilityPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/vendors" element={<VendorsPage />} />
      </Routes>
    </Router>
  );
}

export default App;
