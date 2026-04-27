import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(train_path, test_path):
    return pd.read_csv(train_path), pd.read_csv(test_path)


# -------------------------------
# CLEAN DATA
# -------------------------------
def clean_data(df):
    drop_cols = [
        'Unnamed: 0', 'trans_num', 'first', 'last', 'street',
        'cc_num'   # ID column (must remove)
    ]
    return df.drop(columns=drop_cols, errors='ignore')


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def feature_engineering(df):

    # ---------------------------
    # Transaction datetime
    # ---------------------------
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')

        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day'] = df['trans_date_trans_time'].dt.day
        df['month'] = df['trans_date_trans_time'].dt.month

    # ---------------------------
    # DOB → AGE (CORRECT WAY)
    # ---------------------------
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

        # Use transaction time if available
        if 'trans_date_trans_time' in df.columns:
            ref_time = df['trans_date_trans_time']
        else:
            ref_time = pd.Timestamp.today()

        # Age calculation
        df['age'] = (ref_time - df['dob']).dt.days // 365

        # Handle missing / invalid
        df['age'] = df['age'].fillna(df['age'].median())
        df['age'] = df['age'].clip(lower=18, upper=100)

    # ---------------------------
    # DROP RAW DATE COLUMNS
    # ---------------------------
    df = df.drop(columns=['trans_date_trans_time', 'dob'], errors='ignore')

    return df


# -------------------------------
# ENCODING
# -------------------------------
def encode_data(train_df, test_df):
    categorical_cols = ['category', 'merchant', 'gender', 'city', 'state', 'job']

    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols].astype(str))
    test_df[categorical_cols] = encoder.transform(test_df[categorical_cols].astype(str))

    return train_df, test_df, encoder


# -------------------------------
# SPLIT DATA
# -------------------------------
def split_data(df):
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    return X, y