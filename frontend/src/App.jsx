// App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx"; 
import UploadPage from "./pages/UploadPage.jsx";
import GetStartedPage from "./pages/GetStartedPage";
import LearnMorePage from "./pages/LearnMorePage";
import SustainabilityPage from "./pages/SustainabilityPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />      
        <Route path="/upload" element={<UploadPage />} /> 
        <Route path="/get-started" element={<GetStartedPage />} />
        <Route path="/learn-more" element={<LearnMorePage />} />
        <Route path="/sustainability" element={<SustainabilityPage />} />
      </Routes>
    </Router>
  );
}

export default App;
