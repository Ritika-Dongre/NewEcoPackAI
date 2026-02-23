import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

const VendorsPage = () => {
  const navigate = useNavigate();
  const [vendors, setVendors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchVendors = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/vendors`);
        setVendors(response.data?.items || []);
        setError(null);
      } catch (err) {
        const backendMessage = err?.response?.data?.error || err?.response?.data?.message || null;
        setError(backendMessage || "Failed to load vendors.");
      } finally {
        setLoading(false);
      }
    };
    fetchVendors();
  }, []);

  return (
    <div className="min-h-screen bg-emerald-50 px-4 py-6 sm:px-6 lg:px-10">
      <div className="mx-auto max-w-6xl rounded-3xl border border-emerald-100 bg-white p-5 shadow-sm sm:p-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-emerald-700">EcoPack AI</p>
            <h1 className="text-2xl font-semibold text-emerald-900">Vendors Directory</h1>
          </div>
          <button
            type="button"
            onClick={() => navigate("/upload")}
            className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-sm text-emerald-800 hover:bg-emerald-100"
          >
            Back to Upload
          </button>
        </div>

        {loading && <p className="text-sm text-slate-600">Loading vendors...</p>}
        {error && <p className="rounded border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}

        {!loading && !error && vendors.length === 0 && (
          <p className="text-sm text-slate-600">No vendors found.</p>
        )}

        {!loading && !error && vendors.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {vendors.map((vendor) => (
              <div key={vendor.vendor_id} className="rounded-xl border border-emerald-100 bg-emerald-50/30 p-3">
                <div className="mb-1 flex items-center justify-between">
                  <p className="font-semibold text-emerald-900">{vendor.name}</p>
                  <span className="rounded bg-emerald-100 px-2 py-1 text-xs text-emerald-800">
                    {vendor.rating} / 5
                  </span>
                </div>
                <p className="text-sm text-slate-700">
                  MOQ: {vendor.min_order_qty} | Lead Time: {vendor.lead_time_days} days
                </p>
                <p className="text-sm text-slate-700">Regions: {(vendor.service_regions || []).join(", ")}</p>
                <p className="mt-1 text-sm text-slate-700">Materials: {(vendor.materials || []).join(", ")}</p>
                <div className="mt-2 flex gap-4 text-sm">
                  <a href={vendor.website} target="_blank" rel="noreferrer" className="text-emerald-700 hover:underline">
                    Website
                  </a>
                  <a href={`mailto:${vendor.email}`} className="text-emerald-700 hover:underline">
                    Email
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default VendorsPage;
