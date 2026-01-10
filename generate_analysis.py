import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv('traffic_data.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

# 2. Feature Engineering
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.day_name()
# Create Lag Features (Past traffic predicting future traffic)
df = df.sort_values(by=['Junction', 'DateTime'])
df['Lag_1h'] = df.groupby('Junction')['Vehicles'].shift(1)
df['Lag_24h'] = df.groupby('Junction')['Vehicles'].shift(24)
df_clean = df.dropna()

# 3. Generate Visualizations (Saved as PNGs for your Report)
sns.set_style("whitegrid")

# Plot A: Peak Hour Analysis
plt.figure(figsize=(12, 6))
hourly_traffic = df_clean.groupby('Hour')['Vehicles'].mean()
sns.lineplot(x=hourly_traffic.index, y=hourly_traffic.values, marker='o', color='darkblue', linewidth=2)
plt.title('Average Traffic Volume by Hour (Peak Detection)')
plt.xlabel('Hour of Day')
plt.ylabel('Traffic Volume (Standardized)')
plt.xticks(range(0, 24))
plt.savefig('viz_peak_hours.png')
print("Generated viz_peak_hours.png")

# Plot B: Event Impact
plt.figure(figsize=(12, 6))
top_events = df_clean['Event_Type'].value_counts().head(6).index
df_events = df_clean[df_clean['Event_Type'].isin(top_events)]
sns.barplot(x='Event_Type', y='Vehicles', data=df_events, palette="viridis", ci=None)
plt.title('Traffic Volume during Top Events')
plt.xticks(rotation=15)
plt.savefig('viz_event_impact.png')
print("Generated viz_event_impact.png")

# 4. Train Model
print("Training Model...")
features = ['Hour', 'Temperature_C', 'Precipitation_mm', 'Lag_1h', 'Lag_24h']
X = df_clean[features]
y = df_clean['Vehicles']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"Model Training Complete. R2 Score: {r2:.4f}")

# Save Model
joblib.dump(model, 'traffic_model.pkl')
print("Model saved as traffic_model.pkl")