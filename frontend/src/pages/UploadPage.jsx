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

  // Handle file upload to backend
  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file); // Backend expects "file"

    try {
      const response = await axios.post("http://localhost:5000/classify", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data); // Save backend response
      setError(null);
    } catch (err) {
      console.error("Upload error:", err.response?.data || err.message);
      setError(err.response?.data?.error || "Upload failed");
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
              <h3>Type: {result.product_type}</h3>
              <p>
                <strong>Prediction Accuracy:</strong> {result.prediction_accuracy}
              </p>

              <p>
                <strong>Internal Packaging:</strong> {result.packaging_suggestion?.internal?.material || "N/A"}
              </p>
              <p>
                <em>{result.packaging_suggestion?.internal?.reason || "N/A"}</em>
              </p>

              <p>
                <strong>External Packaging:</strong> {result.packaging_suggestion?.external?.material || "N/A"}
              </p>
              <p>
                <em>{result.packaging_suggestion?.external?.reason || "N/A"}</em>
              </p>

              {result.product_type === "Uncertain" && (
                <p className="warning-text">
                   Model is unsure about this product
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UploadPage;
