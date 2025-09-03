import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to save plots
output_dir = 'plant_doctor_plots'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load the data (with header)
file_path = 'plant_doctor_training_data.csv'
data = pd.read_csv(file_path)

# Optional: Check for any leading/trailing spaces in column names
data.columns = data.columns.str.strip()

print("Data loaded successfully!")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\nFirst 5 rows:")
print(data.head())

# Convert all feature columns to numeric (just in case)
feature_columns = ['moisture', 'temperature', 'ph']
risk_columns = ['pythium_risk', 'phytophthora_risk', 'rhizoctonia_risk', 'southern_blight_risk']

data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors='coerce')

# Drop any corrupted rows (if any)
data.dropna(inplace=True)

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(16, 12))

# --- 1. Distribution of Features ---
plt.subplot(2, 3, 1)
sns.histplot(data['moisture'], kde=True, color='skyblue', bins=30)
plt.title('Distribution of Moisture')

plt.subplot(2, 3, 2)
sns.histplot(data['temperature'], kde=True, color='salmon', bins=30)
plt.title('Distribution of Temperature')

plt.subplot(2, 3, 3)
sns.histplot(data['ph'], kde=True, color='lightgreen', bins=30)
plt.title('Distribution of pH')

# --- 2. Boxplot of Features ---
plt.subplot(2, 3, 4)
sns.boxplot(data=data[feature_columns])
plt.title('Boxplot of Features')

# --- 3. Correlation Heatmap ---
plt.subplot(2, 3, 5)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')

# --- 4. Scatter: Moisture vs Temperature, colored by southern_blight_risk ---
plt.subplot(2, 3, 6)
scatter = plt.scatter(data['moisture'], data['temperature'], c=data['southern_blight_risk'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Southern Blight Risk')
plt.xlabel('Moisture')
plt.ylabel('Temperature')
plt.title('Moisture vs Temperature (colored by Risk)')

# Save the first figure
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overview_plots.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- 5. Pairplot ---
pairplot_fig = sns.pairplot(data[feature_columns + ['southern_blight_risk']], hue='southern_blight_risk', palette='viridis', plot_kws={'alpha': 0.7})
pairplot_fig.fig.suptitle('Pairwise Relationships by Southern Blight Risk', y=1.02, fontsize=16)
pairplot_fig.savefig(os.path.join(output_dir, 'pairplot.png'), dpi=300, bbox_inches='tight')

# --- 6. Risk Condition Counts ---
plt.figure(figsize=(10, 6))
risk_data = data[risk_columns].sum().reset_index()
risk_data.columns = ['Disease', 'Count']
sns.barplot(data=risk_data, x='Disease', y='Count', palette='Set2')
plt.title('Count of High Risk Conditions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'risk_counts.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- 7. pH vs Moisture colored by Risk ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data['moisture'], data['ph'], c=data['southern_blight_risk'], cmap='RdYlGn_r', alpha=0.7)
plt.colorbar(scatter, label='Southern Blight Risk')
plt.xlabel('Moisture')
plt.ylabel('pH')
plt.title('Moisture vs pH (colored by Risk Level)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'moisture_vs_pH_risk.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ All plots saved in folder: '{output_dir}'")