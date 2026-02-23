import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

const HistoryPage = () => {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [summary, setSummary] = useState(null);
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const [historyRes, summaryRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/history?limit=100`),
          axios.get(`${API_BASE_URL}/history/summary`),
        ]);
        setItems(historyRes.data?.items || []);
        setSummary(summaryRes.data || null);
        setError(null);
      } catch (err) {
        const backendMessage = err?.response?.data?.error || err?.response?.data?.message || null;
        setError(backendMessage || "Failed to load history.");
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      if (filter === "all") return true;
      if (filter === "uncertain") return item.product_type === "Uncertain";
      if (filter === "high_confidence") return (item.prediction_accuracy || 0) >= 0.8;
      return true;
    });
  }, [items, filter]);

  return (
    <div className="min-h-screen bg-emerald-50 px-4 py-6 sm:px-6 lg:px-10">
      <div className="mx-auto max-w-6xl rounded-3xl border border-emerald-100 bg-white p-5 shadow-sm sm:p-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-emerald-700">EcoPack AI</p>
            <h1 className="text-2xl font-semibold text-emerald-900">Prediction History</h1>
          </div>
          <button
            type="button"
            onClick={() => navigate("/upload")}
            className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-sm text-emerald-800 hover:bg-emerald-100"
          >
            Back to Upload
          </button>
        </div>

        {summary && (
          <p className="mb-3 text-sm text-slate-600">
            Total: {summary.total_predictions} | Avg Confidence:{" "}
            {((summary.average_confidence || 0) * 100).toFixed(1)}% | Uncertain: {summary.uncertain_predictions}
          </p>
        )}

        <div className="mb-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setFilter("all")}
            className={`rounded px-3 py-1.5 text-xs ${
              filter === "all" ? "bg-emerald-700 text-white" : "bg-emerald-100 text-emerald-800"
            }`}
          >
            All
          </button>
          <button
            type="button"
            onClick={() => setFilter("uncertain")}
            className={`rounded px-3 py-1.5 text-xs ${
              filter === "uncertain" ? "bg-amber-500 text-white" : "bg-amber-100 text-amber-800"
            }`}
          >
            Uncertain
          </button>
          <button
            type="button"
            onClick={() => setFilter("high_confidence")}
            className={`rounded px-3 py-1.5 text-xs ${
              filter === "high_confidence" ? "bg-cyan-700 text-white" : "bg-cyan-100 text-cyan-800"
            }`}
          >
            High Confidence
          </button>
        </div>

        {loading && <p className="text-sm text-slate-600">Loading history...</p>}
        {error && <p className="rounded border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}

        {!loading && !error && filteredItems.length === 0 && (
          <p className="text-sm text-slate-600">No predictions found.</p>
        )}

        {!loading && !error && filteredItems.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {filteredItems.map((item, idx) => (
              <div key={`${item.timestamp_utc}-${idx}`} className="rounded-xl border border-emerald-100 bg-emerald-50/30 p-3">
                <p className="text-base font-semibold text-emerald-900">
                  {item.product_type} ({((item.prediction_accuracy || 0) * 100).toFixed(1)}%)
                </p>
                <p className="text-sm text-slate-600">File: {item.uploaded_file || "N/A"}</p>
                <p className="text-xs text-slate-500">
                  {item.timestamp_utc ? new Date(item.timestamp_utc).toLocaleString() : "-"}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default HistoryPage;
