# CO2-Concentration-Time-Series-Forecasting
A time series forecasting project modeling atmospheric CO2 using polynomial regression and seasonal decomposition. It fits linear, quadratic, and cubic trends, adds monthly periodic effects, and evaluates model accuracy with RMSE and MAPE. Final model captures both long-term and seasonal patterns.

📈 Time Series Forecasting of CO₂ Concentrations using Polynomial Trend and Seasonal Decomposition
This project, created for the MIT IDS.147[J] course Statistical Machine Learning and Data Science, explores a structured approach to modeling atmospheric CO₂ concentration data through time series analysis and polynomial regression. The goal is to develop an interpretable forecasting model that captures both long-term trends and recurring seasonal patterns.

🔍 Project Overview
The dataset contains monthly CO₂ measurements over several decades. The project includes the following key steps:

Data Cleaning and Preprocessing:
The script loads raw CO₂ data, filters out irrelevant columns, handles missing values (-99.99), and generates a normalized time index.

Trend Modeling:
A series of polynomial regression models (linear, quadratic, cubic) are trained to model the long-term growth trend in CO₂.
Residuals and prediction errors (RMSE, MAPE) are used to compare model performance.
The quadratic model is selected based on its balanced error metrics and residual behavior.

Seasonal Decomposition:
Residuals from the quadratic model are grouped by month to extract the average seasonal component (P_i), capturing yearly periodic fluctuations.
A final model F(t) + P_i is constructed by adding the seasonal signal to the quadratic trend.

Evaluation and Visualization:
The model is validated against a held-out test set.
RMSE and MAPE are computed before and after adding the periodic signal.
Several plots are generated to visualize the raw data, fitted trends, residuals, and final model predictions.

Amplitude Analysis:
Quantifies the amplitudes of the trend, periodic signal, and residual noise.
Computes amplitude ratios to assess the model’s ability to explain variability in the data.

📊 Key Outputs
Final model: Quadratic trend + monthly seasonal component
Forecast accuracy: RMSE and MAPE significantly improve after adding the seasonal signal.

Visuals include:
Trend fits
Residual plots
Final forecast vs actual CO₂ concentrations
Periodic signal strength per month

⚙️ Technologies Used
Python (NumPy, Pandas, scikit-learn, Matplotlib)
Linear and polynomial regression
Residual-based seasonal decomposition
Forecast accuracy metrics (RMSE, MAPE)

📁 File Structure
CO2.csv: Input dataset (from MIT course release)

Main script: Performs end-to-end modeling from preprocessing to forecast evaluation

Special thanks to Dr. Uhler and Dr. Jegelka for compling this project!
