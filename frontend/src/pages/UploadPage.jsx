// UploadPage.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

const UploadPage = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [correctLabel, setCorrectLabel] = useState("");
  const [feedbackNote, setFeedbackNote] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState("");
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [weightKg, setWeightKg] = useState("");
  const [fragile, setFragile] = useState("no");
  const [moistureSensitive, setMoistureSensitive] = useState("no");
  const [shippingDistance, setShippingDistance] = useState("local");
  const [budgetPriority, setBudgetPriority] = useState("balanced");
  const [historySummary, setHistorySummary] = useState(null);

  // Handle file selection
  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreviewURL(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
      setShowFeedbackForm(false);
      setCorrectLabel("");
      setFeedbackNote("");
      setFeedbackStatus("");
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
    formData.append("weight_kg", weightKg);
    formData.append("fragile", fragile === "yes" ? "true" : "false");
    formData.append("moisture_sensitive", moistureSensitive === "yes" ? "true" : "false");
    formData.append("shipping_distance", shippingDistance);
    formData.append("budget_priority", budgetPriority);

    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/classify`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      setError(null);
      setFeedbackStatus("");
      setShowFeedbackForm(false);
      fetchHistory();
    } catch (err) {
      console.error("Upload error:", err);
      const backendMessage =
        err?.response?.data?.error ||
        err?.response?.data?.message ||
        null;
      setError(
        backendMessage ||
          "Backend connection failed. Please make sure Flask is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitFeedback = async () => {
    if (!result) return;

    try {
      setSubmittingFeedback(true);
      setFeedbackStatus("");
      await axios.post(`${API_BASE_URL}/feedback`, {
        predicted_label: result.product_type,
        prediction_accuracy: result.prediction_accuracy,
        correct_label: correctLabel.trim() || null,
        note: feedbackNote.trim() || null,
        uploaded_file: file?.name || null,
        top_predictions: result.top_predictions || [],
      });
      setFeedbackStatus("Thanks! Feedback saved.");
      setShowFeedbackForm(false);
    } catch (err) {
      console.error("Feedback submit error:", err);
      setFeedbackStatus("Feedback save failed. Please try again.");
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const [, summaryRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/history?limit=8`),
        axios.get(`${API_BASE_URL}/history/summary`),
      ]);
      setHistorySummary(summaryRes.data || null);
    } catch (err) {
      console.error("History fetch error:", err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleExportReport = () => {
    if (!result) return;

    const topPredictionsHtml = (result.top_predictions || [])
      .map(
        (item) =>
          `<li>${item.label}: ${(item.confidence * 100).toFixed(1)}%</li>`
      )
      .join("");

    const adjustmentNotesHtml = (result.adjustment_notes || [])
      .map((note) => `<li>${note}</li>`)
      .join("");

    const comparatorHtml = (result.packaging_options || [])
      .map(
        (option) => `
          <tr>
            <td>${option.option_name}${option.is_recommended ? " (Recommended)" : ""}</td>
            <td>${option.scores?.sustainability_score || "-"}/10</td>
            <td>${option.scores?.protection_score || "-"}/10</td>
            <td>${option.scores?.cost_efficiency_score || "-"}/10</td>
            <td>Rs ${option.estimated_cost_inr ?? "-"}</td>
            <td>${option.estimated_co2_g ?? "-"} g</td>
          </tr>
        `
      )
      .join("");

    const reportHtml = `
      <html>
        <head>
          <title>Eco Pack AI Report</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 24px; color: #1f2937; }
            h1, h2 { color: #166534; margin-bottom: 8px; }
            .card { border: 1px solid #d1fae5; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
            table { width: 100%; border-collapse: collapse; margin-top: 8px; }
            th, td { border: 1px solid #d1d5db; padding: 8px; text-align: left; font-size: 12px; }
            th { background: #ecfdf5; }
            ul { margin: 6px 0 0 18px; }
            .meta { color: #4b5563; font-size: 12px; margin-bottom: 12px; }
          </style>
        </head>
        <body>
          <h1>Eco Pack AI Recommendation Report</h1>
          <p class="meta">Generated on: ${new Date().toLocaleString()}</p>

          <div class="card">
            <h2>Prediction</h2>
            <p><strong>Product Type:</strong> ${result.product_type}</p>
            <p><strong>Confidence:</strong> ${(result.prediction_accuracy * 100).toFixed(1)}%</p>
            <p><strong>Threshold:</strong> ${result.confidence_threshold ?? "-"}</p>
            <p><strong>Top Predictions:</strong></p>
            <ul>${topPredictionsHtml || "<li>N/A</li>"}</ul>
          </div>

          <div class="card">
            <h2>Recommended Packaging</h2>
            <p><strong>Internal:</strong> ${result.packaging_suggestion?.internal?.material || "N/A"}</p>
            <p>${result.packaging_suggestion?.internal?.reason || ""}</p>
            <p><strong>External:</strong> ${result.packaging_suggestion?.external?.material || "N/A"}</p>
            <p>${result.packaging_suggestion?.external?.reason || ""}</p>
          </div>

          <div class="card">
            <h2>Score Summary</h2>
            <p><strong>Eco:</strong> ${result.packaging_explanation?.overall?.sustainability_score ?? "-"}/10</p>
            <p><strong>Protection:</strong> ${result.packaging_explanation?.overall?.protection_score ?? "-"}/10</p>
            <p><strong>Cost:</strong> ${result.packaging_explanation?.overall?.cost_efficiency_score ?? "-"}/10</p>
          </div>

          <div class="card">
            <h2>Comparator</h2>
            <table>
              <thead>
                <tr>
                  <th>Option</th>
                  <th>Eco</th>
                  <th>Protection</th>
                  <th>Cost Score</th>
                  <th>Est. Cost</th>
                  <th>Est. CO2</th>
                </tr>
              </thead>
              <tbody>${comparatorHtml || "<tr><td colspan='6'>N/A</td></tr>"}</tbody>
            </table>
          </div>

          <div class="card">
            <h2>Manual Input Adjustments</h2>
            <ul>${adjustmentNotesHtml || "<li>No adjustments applied.</li>"}</ul>
          </div>
        </body>
      </html>
    `;

    const printWindow = window.open("", "_blank", "width=900,height=700");
    if (!printWindow) return;
    printWindow.document.write(reportHtml);
    printWindow.document.close();
    printWindow.focus();
    setTimeout(() => {
      printWindow.print();
    }, 300);
  };

  return (
    <div
      className="min-h-screen px-4 py-6 sm:px-6 lg:px-10"
      style={{
        backgroundImage: "url('/images/background.webp')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        fontFamily: "'Segoe UI', 'Trebuchet MS', sans-serif",
      }}
    >
      <div className="mx-auto max-w-7xl rounded-3xl border border-emerald-100 bg-white/90 p-4 shadow-sm backdrop-blur-md sm:p-6">
        <div className="mb-6 flex flex-col gap-3 text-left sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-emerald-700">EcoPack AI</p>
            <h1 className="mt-1 text-2xl font-semibold text-emerald-900 sm:text-3xl">Packaging Intelligence Workspace</h1>
            <p className="mt-1 text-sm text-slate-600">
              Upload a product image, customize delivery context, and compare packaging strategies.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 self-start">
            <button
              type="button"
              onClick={() => navigate("/vendors")}
              className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800 hover:bg-emerald-100"
            >
              View Vendors
            </button>
            <button
              type="button"
              onClick={() => navigate("/history")}
              className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800 hover:bg-emerald-100"
            >
              View History
            </button>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-12">
          <section className="lg:col-span-4 rounded-2xl border border-emerald-100 bg-white p-4 text-left shadow-sm sm:p-5 lg:sticky lg:top-6 lg:h-fit">
            <h2 className="text-lg font-semibold text-emerald-900">Upload and Context</h2>
            <p className="mt-1 text-sm text-slate-600">Provide input data for better recommendations.</p>

        <label
          htmlFor="fileInput"
          className="mt-4 block cursor-pointer rounded-2xl border-2 border-dashed border-emerald-300 bg-emerald-50/60 p-5 transition hover:border-emerald-500 hover:bg-emerald-50"
        >
          {previewURL ? (
            <img
              src={previewURL}
              alt="Preview"
              className="mx-auto max-h-52 w-full rounded-xl object-contain"
            />
          ) : (
            <p className="rounded-xl bg-white px-3 py-16 text-center text-sm text-slate-600">Drag and drop or click to upload</p>
          )}
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
          />
        </label>

        <div className="mt-4 grid grid-cols-1 gap-3 text-left">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Weight (kg)</label>
            <input
              type="number"
              step="0.1"
              min="0"
              value={weightKg}
              onChange={(e) => setWeightKg(e.target.value)}
              placeholder="e.g., 1.2"
              className="w-full rounded-xl border border-emerald-200 px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-emerald-500"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Fragile</label>
              <select
                value={fragile}
                onChange={(e) => setFragile(e.target.value)}
                className="w-full rounded-xl border border-emerald-200 px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-emerald-500"
              >
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Moisture Sensitive</label>
              <select
                value={moistureSensitive}
                onChange={(e) => setMoistureSensitive(e.target.value)}
                className="w-full rounded-xl border border-emerald-200 px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-emerald-500"
              >
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Shipping</label>
              <select
                value={shippingDistance}
                onChange={(e) => setShippingDistance(e.target.value)}
                className="w-full rounded-xl border border-emerald-200 px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-emerald-500"
              >
                <option value="local">Local</option>
                <option value="regional">Regional</option>
                <option value="long">Long Distance</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Budget Priority</label>
              <select
                value={budgetPriority}
                onChange={(e) => setBudgetPriority(e.target.value)}
                className="w-full rounded-xl border border-emerald-200 px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-emerald-500"
              >
                <option value="balanced">Balanced</option>
                <option value="low_cost">Low Cost</option>
                <option value="eco_first">Eco First</option>
              </select>
            </div>
          </div>
        </div>

        <button
          onClick={handleUpload}
          disabled={loading}
          className={`mt-6 w-full rounded-xl px-4 py-3 text-sm font-semibold text-white transition ${
            loading
              ? "cursor-not-allowed bg-emerald-300"
              : "bg-emerald-700 hover:bg-emerald-800"
          }`}
        >
          {loading ? "Analyzing..." : "Upload and Classify"}
        </button>

        {error && (
          <p className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
            {error}
          </p>
        )}
      </section>

      <section className="lg:col-span-8 space-y-6 text-left">

        {!result && (
          <div className="rounded-2xl border border-dashed border-emerald-200 bg-white/80 p-8 text-center">
            <h3 className="text-xl font-semibold text-emerald-900">No analysis yet</h3>
            <p className="mt-2 text-sm text-slate-600">
              Upload an image on the left and click Upload and Classify to generate recommendations.
            </p>
          </div>
        )}

        {result && (
          <div className="rounded-2xl border border-emerald-100 bg-white p-5 text-left shadow-sm sm:p-6">
            <h3 className="text-2xl font-semibold text-emerald-800 mb-2">
              {result.product_type}
            </h3>
            <p className="text-gray-700 mb-4">
              <strong>Confidence:</strong> {(result.prediction_accuracy * 100).toFixed(1)}%
            </p>
            <button
              type="button"
              onClick={handleExportReport}
              className="mb-4 px-3 py-2 text-sm rounded-lg bg-emerald-600 text-white hover:bg-emerald-700"
            >
              Export Report (Print/PDF)
            </button>
            {result.product_type === "Uncertain" && (
              <p className="text-amber-700 text-sm mb-4">
                Prediction confidence is low. Try a clearer image with better lighting and single product focus.
              </p>
            )}

            {Array.isArray(result.top_predictions) && result.top_predictions.length > 0 && (
              <div className="mb-5">
                <h4 className="font-semibold text-green-700 mb-2">Top Predictions</h4>
                <div className="space-y-2">
                  {result.top_predictions.map((item) => (
                    <div key={item.label}>
                      <div className="flex justify-between text-sm text-gray-700 mb-1">
                        <span>{item.label}</span>
                        <span>{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-green-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-600"
                          style={{ width: `${Math.max(2, item.confidence * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.packaging_explanation?.overall && (
              <div className="mb-5 bg-white border border-green-100 rounded-lg p-3">
                <h4 className="font-semibold text-green-700 mb-2">Why This Packaging</h4>
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div className="bg-green-100 rounded-md p-2">
                    <p className="text-gray-600">Eco</p>
                    <p className="font-semibold text-green-800">
                      {result.packaging_explanation.overall.sustainability_score}/10
                    </p>
                  </div>
                  <div className="bg-emerald-100 rounded-md p-2">
                    <p className="text-gray-600">Protection</p>
                    <p className="font-semibold text-emerald-800">
                      {result.packaging_explanation.overall.protection_score}/10
                    </p>
                  </div>
                  <div className="bg-lime-100 rounded-md p-2">
                    <p className="text-gray-600">Cost</p>
                    <p className="font-semibold text-lime-800">
                      {result.packaging_explanation.overall.cost_efficiency_score}/10
                    </p>
                  </div>
                </div>
              </div>
            )}

            {Array.isArray(result.adjustment_notes) && result.adjustment_notes.length > 0 && (
              <div className="mb-5 bg-amber-50 border border-amber-200 rounded-lg p-3">
                <h4 className="font-semibold text-amber-700 mb-2">Manual Input Adjustments</h4>
                <ul className="text-xs text-amber-800 list-disc pl-4 space-y-1">
                  {result.adjustment_notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </div>
            )}

            {Array.isArray(result.packaging_options) && result.packaging_options.length > 0 && (
              <div className="mb-5">
                <h4 className="font-semibold text-green-700 mb-2">Packaging Comparator</h4>
                <div className="space-y-3">
                  {result.packaging_options.map((option) => (
                    <div
                      key={option.option_name}
                      className={`rounded-lg border p-3 ${
                        option.is_recommended
                          ? "border-green-500 bg-green-50"
                          : "border-green-100 bg-white"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <p className="font-semibold text-sm text-green-800">{option.option_name}</p>
                        {option.is_recommended && (
                          <span className="text-xs px-2 py-1 rounded bg-green-600 text-white">
                            Recommended
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-700 mb-2">
                        Eco {option.scores?.sustainability_score}/10 | Protection{" "}
                        {option.scores?.protection_score}/10 | Cost{" "}
                        {option.scores?.cost_efficiency_score}/10
                      </p>
                      <p className="text-xs text-gray-700 mb-2">
                        Est. Cost: <strong>Rs {option.estimated_cost_inr}</strong> | Est. CO2:{" "}
                        <strong>{option.estimated_co2_g} g</strong>
                      </p>
                      <p className="text-xs text-gray-700">
                        <strong>Internal:</strong> {option.packaging_suggestion?.internal?.material}
                      </p>
                      <p className="text-xs text-gray-700">
                        <strong>External:</strong> {option.packaging_suggestion?.external?.material}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {Array.isArray(result.vendor_options) && result.vendor_options.length > 0 && (
              <div className="mb-5">
                <div className="flex items-center justify-between gap-2">
                  <h4 className="font-semibold text-green-700">Available Vendors</h4>
                  <button
                    type="button"
                    onClick={() => navigate("/vendors")}
                    className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800 hover:bg-emerald-100"
                  >
                    View Vendors ({result.vendor_options.length})
                  </button>
                </div>
                <p className="mt-2 text-xs text-gray-600">
                  Open the Vendors page to browse all details in a dedicated view.
                </p>
              </div>
            )}

            <div className="mb-4">
              <h4 className="font-semibold text-green-700">Internal Packaging</h4>
              <p>{result.packaging_suggestion?.internal?.material}</p>
              <em className="text-gray-500">
                {result.packaging_suggestion?.internal?.reason}
              </em>
              {result.packaging_explanation?.internal && (
                <p className="text-xs text-gray-600 mt-2">
                  Eco {result.packaging_explanation.internal.sustainability_score}/10 | Protection{" "}
                  {result.packaging_explanation.internal.protection_score}/10 | Cost{" "}
                  {result.packaging_explanation.internal.cost_efficiency_score}/10
                </p>
              )}
            </div>

            <div>
              <h4 className="font-semibold text-green-700">External Packaging</h4>
              <p>{result.packaging_suggestion?.external?.material}</p>
              <em className="text-gray-500">
                {result.packaging_suggestion?.external?.reason}
              </em>
              {result.packaging_explanation?.external && (
                <p className="text-xs text-gray-600 mt-2">
                  Eco {result.packaging_explanation.external.sustainability_score}/10 | Protection{" "}
                  {result.packaging_explanation.external.protection_score}/10 | Cost{" "}
                  {result.packaging_explanation.external.cost_efficiency_score}/10
                </p>
              )}
            </div>

            <div className="mt-6 border-t border-green-100 pt-4">
              <h4 className="font-semibold text-green-700 mb-2">Prediction Feedback</h4>
              <div className="flex gap-2 flex-wrap">
                <button
                  type="button"
                  onClick={() => {
                    setShowFeedbackForm(false);
                    setFeedbackStatus("Great! Marked as correct.");
                  }}
                  className="px-3 py-2 text-sm rounded-lg bg-green-600 text-white hover:bg-green-700"
                >
                  Looks Correct
                </button>
                <button
                  type="button"
                  onClick={() => setShowFeedbackForm((prev) => !prev)}
                  className="px-3 py-2 text-sm rounded-lg bg-amber-500 text-white hover:bg-amber-600"
                >
                  Prediction Wrong?
                </button>
              </div>

              {showFeedbackForm && (
                <div className="mt-3 space-y-2">
                  <input
                    type="text"
                    value={correctLabel}
                    onChange={(e) => setCorrectLabel(e.target.value)}
                    placeholder="Correct product type (e.g., Necklace)"
                    className="w-full border border-green-200 rounded-lg p-2 text-sm"
                  />
                  <textarea
                    value={feedbackNote}
                    onChange={(e) => setFeedbackNote(e.target.value)}
                    placeholder="Optional note (why prediction was wrong)"
                    className="w-full border border-green-200 rounded-lg p-2 text-sm"
                    rows={3}
                  />
                  <button
                    type="button"
                    onClick={handleSubmitFeedback}
                    disabled={submittingFeedback}
                    className={`px-3 py-2 text-sm rounded-lg text-white ${
                      submittingFeedback ? "bg-gray-400" : "bg-green-600 hover:bg-green-700"
                    }`}
                  >
                    {submittingFeedback ? "Submitting..." : "Submit Feedback"}
                  </button>
                </div>
              )}

              {feedbackStatus && (
                <p className="text-sm text-gray-700 mt-2">{feedbackStatus}</p>
              )}
            </div>
          </div>
        )}

        <div className="rounded-2xl border border-emerald-100 bg-white p-5 text-left shadow-sm sm:p-6">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-xl font-semibold text-emerald-800">Recent Prediction History</h3>
            <button
              type="button"
              onClick={() => navigate("/history")}
              className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800 hover:bg-emerald-100"
            >
              View History
            </button>
          </div>
          {historySummary && (
            <p className="text-xs text-gray-600 mt-2">
              Total: {historySummary.total_predictions} | Avg Confidence:{" "}
              {((historySummary.average_confidence || 0) * 100).toFixed(1)}% | Uncertain:{" "}
              {historySummary.uncertain_predictions}
            </p>
          )}
          <p className="mt-2 text-xs text-gray-600">
            Open the History page for filters and full prediction records.
          </p>
        </div>
      </section>
    </div>
  </div>
    </div>
  );
};

export default UploadPage;
