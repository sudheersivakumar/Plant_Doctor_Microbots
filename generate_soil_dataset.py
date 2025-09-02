# generate_soil_dataset.py
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of rows
n = 2000

# Generate synthetic data (without timestamp)
data = {
    'moisture': np.random.uniform(20.0, 100.0, n),     # 20–100%
    'temperature': np.random.uniform(15.0, 40.0, n),   # 15–40°C
    'ph': np.random.uniform(4.5, 8.0, n)              # pH 4.5–8.0 (Indian soil range)
}

df = pd.DataFrame(data)

# Label each disease based on known agricultural thresholds

# Pythium Root Rot: Cool, wet, slightly acidic
df['pythium_risk'] = (
    (df['moisture'] > 80) & 
    (df['temperature'] >= 15) & (df['temperature'] <= 25) & 
    (df['ph'] >= 5.0) & (df['ph'] <= 6.5)
).astype(int)

# Phytophthora Root Rot: Warm, wet, acidic
df['phytophthora_risk'] = (
    (df['moisture'] > 75) & 
    (df['temperature'] >= 25) & (df['temperature'] <= 30) & 
    (df['ph'] >= 5.5) & (df['ph'] <= 6.5)
).astype(int)

# Rhizoctonia Damping-Off: Moderate moisture, warm, neutral pH
df['rhizoctonia_risk'] = (
    (df['moisture'] >= 60) & (df['moisture'] <= 80) & 
    (df['temperature'] >= 24) & (df['temperature'] <= 30) & 
    (df['ph'] >= 6.0) & (df['ph'] <= 7.5)
).astype(int)

# Southern Blight: Hot, humid, acidic soil
df['southern_blight_risk'] = (
    (df['moisture'] > 70) & 
    (df['temperature'] > 30) & 
    (df['ph'] < 6.0)
).astype(int)

# Save to CSV
df.to_csv('plant_doctor_training_data.csv', index=False)

# Print summary
print("✅ Dataset generated successfully!")
print(f"📊 Shape: {df.shape}")
print(f"🔍 Dataset Info:")
print(df.info())
print("\n🔍 Disease Prevalence (Risk = 1):")
print(f"   Pythium Risk: {df['pythium_risk'].mean():.2%}")
print(f"   Phytophthora Risk: {df['phytophthora_risk'].mean():.2%}")
print(f"   Rhizoctonia Risk: {df['rhizoctonia_risk'].mean():.2%}")
print(f"   Southern Blight Risk: {df['southern_blight_risk'].mean():.2%}")
print("\n📄 First 5 rows:")
print(df.head())