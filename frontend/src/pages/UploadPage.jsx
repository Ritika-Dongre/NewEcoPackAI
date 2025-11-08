// UploadPage.jsx
import React, { useState } from "react";
import axios from "axios";


const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);  const [loading, setLoading] = useState(false);

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
      setLoading(true);
      const response = await axios.post("http://127.0.0.1:5000/classify", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      setError(null);
    } catch (err) {
      console.error("Upload error:", err);
      setError("Backend connection failed. Please make sure Flask is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-green-50 p-6">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md text-center transition-all hover:shadow-2xl">
        <h1 className="text-3xl font-bold text-green-700 mb-1">Eco Pack AI</h1>
        <p className="text-gray-600 mb-6">Sustainable Packaging Suggestions</p>

        <label
          htmlFor="fileInput"
          className="block border-2 border-dashed border-green-400 rounded-xl p-6 cursor-pointer hover:bg-green-100 transition"
        >
          {previewURL ? (
            <img
              src={previewURL}
              alt="Preview"
              className="rounded-lg mx-auto max-h-48 object-contain"
            />
          ) : (
            <p className="text-gray-500">Drag & drop or click to upload</p>
          )}
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
          />
        </label>

        <button
          onClick={handleUpload}
          disabled={loading}
          className={`mt-6 px-6 py-2 rounded-xl text-white font-medium transition-all duration-300 ${
            loading
              ? "bg-green-400 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading ? "Analyzing..." : "Upload & Classify"}
        </button>

        {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}

        {result && (
          <div className="mt-8 bg-green-50 rounded-xl p-6 text-left shadow-inner">
            <h3 className="text-xl font-semibold text-green-700 mb-2">
              {result.product_type}
            </h3>
            <p className="text-gray-700 mb-4">
              <strong>Accuracy:</strong> {result.prediction_accuracy}
            </p>

            <div className="mb-4">
              <h4 className="font-semibold text-green-700">Internal Packaging</h4>
              <p>{result.packaging_suggestion?.internal?.material}</p>
              <em className="text-gray-500">
                {result.packaging_suggestion?.internal?.reason}
              </em>
            </div>

            <div>
              <h4 className="font-semibold text-green-700">External Packaging</h4>
              <p>{result.packaging_suggestion?.external?.material}</p>
              <em className="text-gray-500">
                {result.packaging_suggestion?.external?.reason}
              </em>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPage;
