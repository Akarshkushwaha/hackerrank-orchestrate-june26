import os
import pandas as pd
from pathlib import Path
from utils import load_data
from processor import ClaimsProcessor
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # Check if we have the Groq API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not found.")
        print("Please set it in your environment or a .env file to run this script.")
        return
        
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "dataset"
    
    print(f"Loading data from {dataset_dir}...")
    claims_df, sample_claims_df, history_df, req_df = load_data(dataset_dir)
    
    if claims_df is None:
        print("Error: Could not load claims.csv")
        return
        
    processor = ClaimsProcessor()
    
    results = []
    total_claims = len(claims_df)
    
    print(f"Processing {total_claims} claims...")
    
    # Order of required columns in output.csv
    output_columns = [
        "user_id", "image_paths", "user_claim", "claim_object", 
        "evidence_standard_met", "evidence_standard_met_reason", 
        "risk_flags", "issue_type", "object_part", "claim_status", 
        "claim_status_justification", "supporting_image_ids", 
        "valid_image", "severity"
    ]
    
    for i, row in claims_df.iterrows():
        print(f"Processing row {i+1}/{total_claims}: user_id {row['user_id']}")
        result = processor.process_claim(row, history_df, req_df, base_dir=base_dir)
        
        # Ensure all columns exist and format correctly
        formatted_result = {col: str(result.get(col, '')).lower() if isinstance(result.get(col), bool) else result.get(col, '') for col in output_columns}
        
        results.append(formatted_result)
        
    output_df = pd.DataFrame(results, columns=output_columns)
    output_path = dataset_dir / "output.csv"
    output_df.to_csv(output_path, index=False)
    
    print(f"\nCompleted! Results saved to {output_path}")
    print(f"Total API Calls: {processor.total_calls}")
    print(f"Total Input Tokens: {processor.total_input_tokens}")
    print(f"Total Output Tokens: {processor.total_output_tokens}")
    
if __name__ == "__main__":
    main()
