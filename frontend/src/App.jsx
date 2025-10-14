// App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx"; 
import UploadPage from "./pages/UploadPage.jsx";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />      
        <Route path="/upload" element={<UploadPage />} /> 
      </Routes>
    </Router>
  );
}

export default App;
