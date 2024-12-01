import React, { useState } from 'react';
import Papa from 'papaparse';
import './Forecast.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import * as tf from '@tensorflow/tfjs';

const Forecast = () => {
  const [data, setData] = useState([]);
  const [months, setMonths] = useState(6);  // Default to 6 months
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState('');
  const [monthsError, setMonthsError] = useState('');  // To hold error message for months input

  // Handle CSV file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        complete: (result) => {
          const processedData = preprocessData(result.data);
          setData(processedData);
        },
        header: true,
      });
    }
  };

  // Data preprocessing function
  const preprocessData = (rawData) => {
    const limitData = Object.values(rawData).slice(0, 100);
    const filteredData = limitData.filter(
      (row) => row.created && row.short_desc && row.total_sold && isNaN(row.short_desc)
    );
    const result = filteredData
      .map((entry) => {
        if (!entry.created || !entry.short_desc || !entry.total_sold) return null;
        const dateParts = entry.created.split('-');
        if (dateParts.length < 2) return null;
        const month = parseInt(dateParts[1], 10);
        return { month, product: entry.short_desc, quantity: parseFloat(entry.total_sold) };
      })
      .filter((entry) => entry !== null);

    return result;
  };

  // Prepare data for model
  const prepareData = (productData) => {
    const months = productData.map((entry) => entry.month);
    const quantities = productData.map((entry) => entry.quantity);

    const xs = tf.tensor2d(months.map((month, index) => [month, index]));
    const ys = tf.tensor1d(quantities);

    return { xs, ys };
  };

  // Function to predict sales for a specific product
  const predictSalesForProduct = async (productData) => {
    const { xs, ys } = prepareData(productData);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, inputShape: [2], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

    await model.fit(xs, ys, { epochs: 100 });

    const predictions = [];
    const lastMonth = productData[productData.length - 1].month;

    for (let i = 1; i <= months; i++) {
      const inputTensor = tf.tensor2d([[lastMonth + i, productData.length]]);
      const predictedQuantity = model.predict(inputTensor).dataSync()[0];
      predictions.push({ month: lastMonth + i, predicted: predictedQuantity });
    }

    return predictions;
  };

  // Function to predict sales for all products
  const predictSales = async () => {
    if (months <= 0) {
      setMonthsError('Please enter a positive number for months.');
      return;
    } else {
      setMonthsError('');
    }
    setLoading(true);

    const productPredictions = {};

    const uniqueProducts = [...new Set(data.map((entry) => entry.product))];

    for (let product of uniqueProducts) {
      const filteredData = data.filter((entry) => entry.product === product);

      if (filteredData.length === 0) {
        alert(`No data found for the product: ${product}`);
        continue;
      }

      const predictions = await predictSalesForProduct(filteredData);
      productPredictions[product] = predictions;
    }

    setPredictions(productPredictions);
    setLoading(false);
  };

  // Handle product selection
  const handleProductChange = (event) => {
    setSelectedProduct(event.target.value);
  };

  // Prepare chart data
  const renderChartData = (product) => {
    const chartData = [];
    const filteredData = data.filter((entry) => entry.product === product);

    filteredData.forEach((entry) => {
      const predictedEntry = predictions[product]?.find((pred) => pred.month === entry.month);
      chartData.push({
        month: entry.month,
        actual: entry.quantity,
        predicted: predictedEntry ? predictedEntry.predicted : null,
      });
    });

    predictions[product]?.forEach((pred) => {
      chartData.push({
        month: pred.month,
        actual: null,
        predicted: pred.predicted,
      });
    });

    return chartData;
  };

  return (
    <div className="forecast-container">
      <h1>Sales Forecast</h1>

      <div className="product-selector">
        <label htmlFor="product">Select Product: </label>
        <select id="product" value={selectedProduct} onChange={handleProductChange}>
          <option value="">Select a Product</option>
          {[...new Set(data.map((entry) => entry.product))].map((product) => (
            <option key={product} value={product}>
              {product}
            </option>
          ))}
        </select>
      </div>

      <div className="file-upload">
        <input type="file" accept=".csv" onChange={handleFileUpload} />
      </div>

      {/* Dynamic months input with improved design */}
      {/* <div className="months-selector">
        <label htmlFor="months">Number of months to predict: </label>
        <input
          type="number"
          id="months"
          value={months}
          onChange={(e) => {
            const value = Number(e.target.value);
            setMonths(value);
            if (value <= 0) setMonthsError('Please enter a positive number for months.');
            else setMonthsError('');
          }}
          min="1"
          style={{ padding: '8px', fontSize: '16px', width: '120px', marginBottom: '10px' }}
        />
        {monthsError && <p style={{ color: 'red', fontSize: '14px' }}>{monthsError}</p>}
      </div> */}

      <button onClick={predictSales} disabled={loading} style={{ padding: '10px 20px', fontSize: '16px' }}>
        Predict Sales
      </button>

      {loading && <div>Loading... Model is training and predicting sales...</div>}

      {selectedProduct && predictions[selectedProduct] && (
        <>
          <h2>Sales Forecast for {selectedProduct}</h2>

          <div style={{ marginTop: '40px' }} />

          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={renderChartData(selectedProduct)} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" label={{ value: 'Month', position: 'bottom', offset: 20 }} />
              <YAxis label={{ value: 'Quantity', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(value) => `${value} units`} />
              <Legend wrapperStyle={{ paddingTop: 30 }} />
              <Line type="monotone" dataKey="actual" stroke="#8884d8" />
              <Line type="monotone" dataKey="predicted" stroke="#82ca9d" />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {/* Display charts for all products */}
      {!selectedProduct && Object.keys(predictions).length > 0 && (
        <>
          <h2>Sales Forecast for All Products</h2>
          {Object.keys(predictions).map((product) => (
            <div key={product} style={{ marginBottom: '40px' }}>
              <h3>{product}</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={renderChartData(product)} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" label={{ value: 'Month', position: 'bottom', offset: 20 }} />
                  <YAxis label={{ value: 'Quantity', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value) => `${value} units`} />
                  <Legend wrapperStyle={{ paddingTop: 30 }} />
                  <Line type="monotone" dataKey="actual" stroke="#8884d8" />
                  <Line type="monotone" dataKey="predicted" stroke="#82ca9d" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </>
      )}
    </div>
  );
};

export default Forecast;
