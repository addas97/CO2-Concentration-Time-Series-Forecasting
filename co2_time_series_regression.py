# Akash Das
# MIT IDS.147[J] Statistical Machine Learning and Data Science
# Module 4: Time Series Forecasting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# == Load Data == 
cols = ["year", "month", "exceldata", "date2", "CO2", "co2_season_adj", "co2_spline_season_adj", "co2_spline", "co2_fill_7", "co2_fill_8"]
data = pd.read_csv('Time_Series_Module4/release_time_series_report_data_nops/CO2.csv', skiprows=57, header=None, names=cols)

# == Clean Data == 

# Drop irrelevant columns
mandatory_cols = ['year', 'month', 'CO2']
for col in cols:
    if col not in mandatory_cols:
        data = data.drop(col, axis = 1)

# Normalize data
data['CO2'] = data['CO2'].replace(-99.99, np.nan)
data['time'] = [(i + 0.5) / 12 for i in range(0, len(data['CO2']))]
data = data.dropna()

# == Linear Modeling == 
time = data['time'].to_list()
co2 = data['CO2'].to_list()
training_size = int(len(time) * 0.8)
testing_size = len(time) - training_size

# Splitting data
x_train = time[:training_size]
y_train = co2[:training_size]
x_test = time[training_size:]
y_test = co2[training_size:]

# Plot raw data
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train)
plt.xlabel('time')
plt.ylabel('CO2 Concentration')
plt.title = "C02 Variation with time"

# Fit linear model
model = LinearRegression()
model.fit(np.array(x_train).reshape(-1, 1), y_train) # we reshape since 1D list
coef = [model.coef_[0], model.intercept_]
y_train_preds = model.predict(np.array(x_train).reshape(-1, 1)) # Draw linear trend on trained data

# Plot linear fit
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train)
#plt.plot(x_train, linear_fit)
plt.xlabel('time')
plt.ylabel('CO2 Concentration')
plt.title = "C02 Variation with time"

# Find and plot residuals 
plt.figure(figsize=(6, 4))
residuals = y_train - y_train_preds
plt.scatter(x_train, residuals)
plt.xlabel('time')
plt.ylabel('Residual')
#plt.show()

# Find RMSE and MAPE
def rmse(y_test, y_preds):
    return np.sqrt(mean_squared_error(y_test, y_preds))

y_test_preds = model.predict(np.array(x_test).reshape(-1, 1))
rmse_linear = rmse(y_test, y_test_preds)
mape_linear = mean_absolute_percentage_error(y_test, y_test_preds) * 100
print(f"Linear RMSE: {rmse_linear:.3f}")
print(f"Linear MAPE: {mape_linear:.3f}%")

'''
Reflection:
So far, based on the shape of the original data and the plot of the residuals, we will now fit 
a quadratic and cubic model to find the best fit for the data
'''

# == Quadratic and Cubic Modeling == 
def print_polynomial_model(model, degree, label="Model"):
    coefs = model.named_steps.linearregression.coef_
    intercept = model.named_steps.linearregression.intercept_
    
    terms = [f"{coef:.4f}*t^{i}" for i, coef in enumerate(coefs[1:], start=1)]  # skip the first coef (bias for x^0)
    equation = " + ".join(reversed(terms))  # highest degree term first
    equation += f" + ({intercept:.4f})"
    
    print(f"The {label} (degree {degree}) is: F(t) = {equation}")

# Degree 2: Quadratic
degree = 2
quad_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
quad_model.fit(np.array(x_train).reshape(-1, 1), y_train)
y_train_preds_quad = quad_model.predict(np.array(x_train).reshape(-1, 1))
print_polynomial_model(quad_model, degree, label="Quadratic Model")

# Degree 3: Cubic
degree = 3
cubic_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
cubic_model.fit(np.array(x_train).reshape(-1, 1), y_train)
y_train_preds_cubic = cubic_model.predict(np.array(x_train).reshape(-1, 1))
print_polynomial_model(cubic_model, degree, label="Cubic Model")

# Plot Quadratic Residuals
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train - y_train_preds_quad)
plt.xlabel('time')
plt.ylabel('Residual')
#plt.show()

# Plot Cubic Residuals
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train - y_train_preds_cubic)
plt.xlabel('time')
plt.ylabel('Residual')
#plt.show()

# Predict: Quadratic
y_test_preds_quad = quad_model.predict(np.array(x_test).reshape(-1, 1))

# Predict: Cubic
y_test_preds_cubic = cubic_model.predict(np.array(x_test).reshape(-1, 1))

# Find RMSE and MAPE: Quadratic
quad_rmse = rmse(y_test, y_test_preds_quad)
quad_mape = mean_absolute_percentage_error(y_test, y_test_preds_quad)
print(f"Quadratic RMSE: {quad_rmse:.3f}")
print(f"Quadratic MAPE: {quad_mape:.3f}%")

# Find RMSE and MAPE: Cubic
cubic_rmse = rmse(y_test, y_test_preds_cubic)
cubic_mape = mean_absolute_percentage_error(y_test, y_test_preds_cubic)
print(f"Cubic RMSE: {cubic_rmse:.3f}")
print(f"Cubic MAPE: {cubic_mape:.3f}%")

'''
Reflection:
Based on the residual plots drawn above and the prediction errors found for the linear, quadratic, and cubic models, the quadratic
model serves as the best fit for the data.
'''

# == Time Series Modeling: Fitting a Periodic Signal == 
data['Quad_preds'] = quad_model.predict(np.array(data['time']).reshape(-1, 1)) # Predict across all times - inclusive of training and testing samples
data['residual'] = data['CO2'] - data['Quad_preds']
monthly_avg_residuals = data.groupby('month')['residual'].mean()

plt.figure(figsize=(6,4))
plt.plot(monthly_avg_residuals)
plt.title('Periodic Signal P_i')
plt.ylabel('Periodicty in CO2 Concentration')
plt.xlabel('Month (i)')
#plt.show()

P1 = monthly_avg_residuals.loc[1]  # January
P2 = monthly_avg_residuals.loc[2]  # February

print(f"Periodic component P1 (Jan): {P1:.2f}")
print(f"Periodic component P2 (Feb): {P2:.2f}")

# Find Final Fit
data['P_i'] = data['month'].map(monthly_avg_residuals) # Match monthly_avg_residual to month
data['Final_Fit'] = data['Quad_preds'] + data['P_i']

final_training = data['Final_Fit'][:training_size]
final_testing = data['Final_Fit'][training_size:]

plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['CO2'], label='Original CO2', alpha=0.6)
plt.plot(data['time'][:training_size], final_training, label='Final Training Model: F_n(t) + P_i', color = 'red', alpha=0.6)
plt.plot(data['time'][training_size:], final_testing, label='Final Testing Model: F_n(t) + P_i', color='green')
plt.axvline(x = data['time'].iloc[training_size], color='purple', linestyle='--', label='Train/Test Split') # Training/test line split
plt.xlabel('Time')
plt.ylabel('CO2 Concentration')
plt.title('Final Fit: F_n(t) + P_i over Time')
plt.legend()
#plt.show()

# Find RMSE and MAPE: Final Quadratic
final_quad_rmse = rmse(y_test, final_testing)
final_quad_mape = mean_absolute_percentage_error(y_test, final_testing)
print(f"Quadratic RMSE: {final_quad_rmse:.3f}")
print(f"Quadratic MAPE: {final_quad_mape:.3f}%")

# Amplitude Analysis
F_amp = np.max(y_train_preds_quad) - np.min(y_train_preds_quad)
print(f"Trend Amplitude: {F_amp}")

training_period = data['P_i'][:training_size]
P_amp = np.max(training_period) - np.min(training_period)
print(f"Periodic Amplitude: {P_amp}")

print(f"Amplitude / Period Ratio: {F_amp / P_amp}")

res_amp = np.max(y_train - y_train_preds_quad - training_period) - np.min(y_train - y_train_preds_quad - training_period)
print(f"Residual Amplitude: {res_amp}")

print(f"Period / Residual Ratio: {P_amp / res_amp}")