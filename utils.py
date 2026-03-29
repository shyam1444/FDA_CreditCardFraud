import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data(filepath="creditcard.csv"):
    if not os.path.exists(filepath):
        st.error(f"Dataset not found at {filepath}. Please ensure it is in the same directory.")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    
    # Optional but recommended preprocessing:
    # 1. Impute any missing values (usually creditcard.csv has none, but it's safe)
    if df.isnull().sum().any():
        df = df.dropna() # Alternatively, could use SimpleImputer
        
    # 2. Scale 'Amount' and 'Time' since they are not PCA-transformed like V1-V28.
    # This is especially crucial for K-Means clustering and distance-based tools.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    if 'Amount' in df.columns:
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        
    return df

@st.cache_data
def get_sample(df, frac=0.1, random_state=42):
    """Returns a stratified sample to preserve the fraud ratio."""
    if df.empty:
        return df
    
    # Stratified sampling
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    
    fraud_sample = fraud.sample(frac=frac, random_state=random_state)
    normal_sample = normal.sample(frac=frac, random_state=random_state)
    
    sample_df = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=random_state)
    return sample_df

def prepare_features(df):
    """Separates X and y."""
    if df.empty:
        return None, None
    X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
    y = df['Class']
    return X, y
