import base64
import pandas as pd
from pathlib import Path

def encode_image_base64(image_path):
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

def load_data(dataset_dir):
    """Loads all necessary CSV files."""
    dataset_path = Path(dataset_dir)
    
    # Read CSVs
    try:
        claims_df = pd.read_csv(dataset_path / "claims.csv")
    except FileNotFoundError:
        claims_df = None
        
    try:
        sample_claims_df = pd.read_csv(dataset_path / "sample_claims.csv")
    except FileNotFoundError:
        sample_claims_df = None
        
    history_df = pd.read_csv(dataset_path / "user_history.csv")
    req_df = pd.read_csv(dataset_path / "evidence_requirements.csv")
    
    return claims_df, sample_claims_df, history_df, req_df

def get_user_history(user_id, history_df):
    """Retrieves user history context."""
    user_row = history_df[history_df["user_id"] == user_id]
    if user_row.empty:
        return "No prior claim history."
    
    flags = user_row.iloc[0]["history_flags"]
    summary = user_row.iloc[0]["history_summary"]
    return f"Flags: {flags}\nSummary: {summary}"

def get_evidence_requirements(claim_object, req_df):
    """Retrieves applicable evidence requirements."""
    # Filter for 'all' or the specific object
    applicable_reqs = req_df[req_df["claim_object"].isin(["all", claim_object])]
    
    req_text = ""
    for _, row in applicable_reqs.iterrows():
        req_text += f"- [{row['applies_to']}] {row['minimum_image_evidence']}\n"
    return req_text
