import React from "react";
import { useNavigate } from "react-router-dom";

const products = [
  { name: "Chocolate Box", packaging: ["Biodegradable Paper Box", "Recycled Cardboard"] },
  { name: "Cosmetics Bottle", packaging: ["Glass Bottle", "Aluminum Tube"] },
  { name: "Book", packaging: ["Recycled Paper Wrap", "Cardboard Sleeve"] },
  { name: "Clothing", packaging: ["Reusable Fabric Bag", "Recycled Polybag"] },
  { name: "Electronics", packaging: ["Molded Pulp Tray", "Recycled Cardboard Box"] },
  { name: "Shoes", packaging: ["Recycled Cardboard Shoe Box", "Reusable Fabric Bag"] },
  { name: "Watches", packaging: ["Bamboo Box", "Recycled Cardboard Sleeve"] },
  { name: "Necklace", packaging: ["Recycled Kraft Paper Box", "Reusable Velvet Pouch"] },
  { name: "Rings", packaging: ["Recycled Cardboard Ring Box", "Fabric Pouch"] },
  { name: "Bracelet", packaging: ["Recycled Kraft Paper Box", "Reusable Pouch"] },
];

const SustainabilityPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-green-50 px-8 md:px-20 py-16 relative">
      <h1 className="text-4xl md:text-5xl font-bold text-green-800 mb-8 text-center">
        Sustainable Packaging Recommendations
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {products.map((product, idx) => (
          <div
            key={idx}
            className="bg-white p-6 rounded-3xl shadow-lg hover:shadow-2xl transition transform hover:scale-105"
          >
            <h2 className="text-2xl font-semibold text-green-700 mb-2">{product.name}</h2>
            <h3 className="text-lg font-medium text-green-600 mb-2">Recommended Packaging:</h3>
            <ul className="list-disc ml-6 text-gray-700 space-y-1">
              {product.packaging.map((pack, i) => (
                <li key={i}>{pack}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="flex justify-center mt-12">
        <button
          onClick={() => navigate(-1)}
          className="bg-green-600 text-white px-8 py-3 rounded-lg font-semibold shadow-lg hover:bg-green-700 transform hover:scale-105 transition duration-300"
        >
          ← Back
        </button>
      </div>
    </div>
  );
};

export default SustainabilityPage;
