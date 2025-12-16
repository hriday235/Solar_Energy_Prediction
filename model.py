import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# --- 1. Load and Clean Data ---
print("Loading data...")
df = pd.read_csv('Solar Energy.csv')

# Handle Missing Values
df['Energy Storage System Size (kWac)'] = df['Energy Storage System Size (kWac)'].fillna(0)
df['Developer'] = df['Developer'].fillna('Unknown')
df['Utility'] = df['Utility'].fillna('Unknown')
df['County'] = df['County'].fillna('Unknown')

# Drop messy rows
df.dropna(inplace=True)

# Fix Date
df['Interconnection Date'] = pd.to_datetime(df['Interconnection Date'])
df['Year'] = df['Interconnection Date'].dt.year
df['Month'] = df['Interconnection Date'].dt.month

# Feature Engineering (Efficiency Filter)
df['Efficiency'] = df['Estimated Annual PV Energy Production (kWh)'] / df['Estimated PV System Size (kWdc)']
df = df[(df['Efficiency'] > 500) & (df['Efficiency'] < 2500)]

# --- 2. Encode Variables (The Translators) ---
print("Encoding variables...")
le_utility = LabelEncoder()
le_county = LabelEncoder()
le_developer = LabelEncoder()

# Fit the encoders on the text data so they learn the names
df['Utility'] = le_utility.fit_transform(df['Utility'].astype(str))
df['County'] = le_county.fit_transform(df['County'].astype(str))
df['Developer'] = le_developer.fit_transform(df['Developer'].astype(str))

# --- 3. Prepare Features (X) and Target (y) ---
# We must match the EXACT order used in the App
# Features: [Utility, County, Developer, Size, Battery, Year, Month]
X = df[['Utility', 'County', 'Developer', 'Estimated PV System Size (kWdc)', 
        'Energy Storage System Size (kWac)', 'Year', 'Month']]
y = df['Estimated Annual PV Energy Production (kWh)']

# --- 4. Train the Model (The Brain) ---
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)  # <--- This is the crucial step you were missing!

# --- 5. Save Everything ---
print("Saving files...")
joblib.dump(lr, 'solar_model.pkl')
joblib.dump(le_utility, 'le_utility.pkl')
joblib.dump(le_county, 'le_county.pkl')
joblib.dump(le_developer, 'le_developer.pkl')

print("SUCCESS! All 4 files (solar_model.pkl + 3 encoders) are saved.")