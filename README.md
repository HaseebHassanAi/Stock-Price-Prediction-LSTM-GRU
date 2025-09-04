# Stock Price Prediction Using LSTM and GRU

This project demonstrates how to predict the next day's stock closing price using **deep learning models**: LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit). The models are trained on historical stock price data from **Kaggle** and evaluated by comparing predicted vs actual prices.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Methods](#methods)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Introduction
Stock price prediction is a challenging task due to its sequential and volatile nature. This project uses **LSTM** and **GRU**, which are specialized recurrent neural networks capable of capturing temporal dependencies in sequential data, to predict the next day's closing price of a stock.

---

## Dataset
- The dataset used in this project is from **Kaggle**: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
- It contains historical stock price data with columns:  
  `Date, Open, High, Low, Close, Adj Close, Volume`
- Download the CSV file from Kaggle and place it in your working directory.

---

## Features
- **Preprocessing:** Handle missing values, remove duplicates, and optionally remove outliers.
- **Scaling:** Normalize the data using Min-Max scaling for better neural network performance.
- **Sequence Generation:** Convert historical prices into time-step sequences for model training.
- **Prediction:** Predict the next day's closing price based on the previous `time_step` days.

---

## Methods
- **LSTM Model:** 
  - Three stacked LSTM layers with Dropout for regularization.
  - Dense output layer for predicting closing price.

- **GRU Model:**
  - Two stacked GRU layers with Dropout.
  - Dense output layer for predicting closing price.

- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam (learning rate = 0.001)

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/Stock-Price-Prediction-LSTM-GRU.git
