// UploadPage.jsx
import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Handle file selection
  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreviewURL(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  // Upload to backend
  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/classify",  // ✅ Updated URL
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(response.data);
      setError(null);

    } catch (err) {
      console.error("Upload error:", err);
      setError("Backend connection failed. Is Flask running?");
    }
  };

  return (
    <div className="upload-page">
      <h2>Upload Product Image</h2>

      <input type="file" onChange={handleChange} />
      <br />
      <button onClick={handleUpload}>Upload & Classify</button>

      {error && <p className="error-text">{error}</p>}

      {previewURL && (
        <div className="profile-card">
          <div className="profile-img-wrapper">
            <img src={previewURL} alt="Preview" className="profile-img" />
          </div>

          {result && (
            <div className="profile-details">
              <h3>Product Type: {result.product_type}</h3>
              <p><strong>Accuracy:</strong> {result.prediction_accuracy}</p>

              <h4>Internal Packaging:</h4>
              <p>{result.packaging_suggestion?.internal?.material}</p>
              <em>{result.packaging_suggestion?.internal?.reason}</em>

              <h4>External Packaging:</h4>
              <p>{result.packaging_suggestion?.external?.material}</p>
              <em>{result.packaging_suggestion?.external?.reason}</em>

              {result.product_type === "Uncertain" && (
                <p className="warning-text">⚠ Model not confident</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UploadPage;
