import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("electricity_theft_dataset.csv")  # Ensure the file is present

# Handle Missing Values (if any)
df.fillna(df.median(), inplace=True)

# Normalize Data (Scaling Features)
scaler = MinMaxScaler()
df[['Consumption_kWh', 'Voltage', 'Frequency', 'Peak_Hours_Usage']] = scaler.fit_transform(
    df[['Consumption_kWh', 'Voltage', 'Frequency', 'Peak_Hours_Usage']]
)

# Handle Imbalanced Data using SMOTE
X = df.drop(columns=['Theft_Label'], errors='ignore')  # Prevents KeyError if column is missing
y = df['Theft_Label']  # Target variable

smote = SMOTE(sampling_strategy='auto', random_state=42) # Balances the dataset by making theft cases 50% of normal cases
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new balanced DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['Theft_Label'] = y_resampled

# Save the preprocessed dataset
df_balanced.to_csv("electricity_theft_preprocessed.csv", index=False)

print("Preprocessing Complete! Balanced dataset saved as 'electricity_theft_preprocessed.csv'")
