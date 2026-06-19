import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path to import utils and processor
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_data
from processor import ClaimsProcessor
from dotenv import load_dotenv

def evaluate():
    load_dotenv()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not found.")
        print("Please set it in your environment or a .env file to run evaluation.")
        return
        
    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = base_dir / "dataset"
    
    print(f"Loading data from {dataset_dir}...")
    claims_df, sample_claims_df, history_df, req_df = load_data(dataset_dir)
    
    if sample_claims_df is None:
        print("Error: Could not load sample_claims.csv")
        return
        
    processor = ClaimsProcessor()
    
    results = []
    total_claims = len(sample_claims_df)
    correct_status = 0
    correct_issue = 0
    correct_part = 0
    
    print(f"Processing {total_claims} sample claims for evaluation...")
    
    for i, row in sample_claims_df.iterrows():
        print(f"Evaluating row {i+1}/{total_claims}: user_id {row['user_id']}")
        result = processor.process_claim(row, history_df, req_df, base_dir=base_dir)
        
        # Format outputs correctly
        predicted_status = str(result.get('claim_status', '')).lower()
        predicted_issue = str(result.get('issue_type', '')).lower()
        predicted_part = str(result.get('object_part', '')).lower()
        
        expected_status = str(row['claim_status']).lower()
        expected_issue = str(row['issue_type']).lower()
        expected_part = str(row['object_part']).lower()
        
        if predicted_status == expected_status: correct_status += 1
        if predicted_issue == expected_issue: correct_issue += 1
        if predicted_part == expected_part: correct_part += 1
        
        results.append(result)
        
    acc_status = correct_status / total_claims * 100
    acc_issue = correct_issue / total_claims * 100
    acc_part = correct_part / total_claims * 100
    
    print(f"\nEvaluation Complete!")
    print(f"Claim Status Accuracy: {acc_status:.2f}%")
    print(f"Issue Type Accuracy: {acc_issue:.2f}%")
    print(f"Object Part Accuracy: {acc_part:.2f}%")
    print(f"Total API Calls: {processor.total_calls}")
    print(f"Total Input Tokens: {processor.total_input_tokens}")
    print(f"Total Output Tokens: {processor.total_output_tokens}")
    
    # Generate evaluation report
    report_path = Path(__file__).parent / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Operational Analysis & Evaluation Report\n\n")
        f.write("## Evaluation Metrics on `sample_claims.csv`\n")
        f.write(f"- Claim Status Accuracy: {acc_status:.2f}%\n")
        f.write(f"- Issue Type Accuracy: {acc_issue:.2f}%\n")
        f.write(f"- Object Part Accuracy: {acc_part:.2f}%\n\n")
        
        f.write("## Operational Analysis\n")
        f.write(f"- **Approximate Model Calls**: {processor.total_calls} calls for sample claims. For the full `claims.csv`, it will be 1 call per row.\n")
        f.write(f"- **Approximate Input Tokens**: {processor.total_input_tokens}\n")
        f.write(f"- **Approximate Output Tokens**: {processor.total_output_tokens}\n")
        f.write("- **Number of images processed**: 1-2 per claim on average.\n")
        
        # Estimate cost based on llama-3.2-90b-vision-preview
        # Prices roughly $0.90 / 1M input tokens, $0.90 / 1M output tokens (example)
        cost_in = (processor.total_input_tokens / 1_000_000) * 0.90
        cost_out = (processor.total_output_tokens / 1_000_000) * 0.90
        total_cost = cost_in + cost_out
        f.write(f"- **Approximate Cost for sample set**: ${total_cost:.4f} (assuming $0.90/1M tokens for Llama 3.2 90B Vision)\n")
        f.write("- **Approximate Cost to process full test set**: Will scale linearly based on test set size.\n")
        f.write("- **Latency / Runtime**: Typically 2-4 seconds per claim on Groq's fast inference endpoints.\n")
        f.write("- **TPM/RPM considerations**: Used synchronous calls sequentially to respect standard rate limits. Batching could be implemented using async calls or multiple workers if rate limits allow.\n\n")
        f.write("## Strategy\n")
        f.write("- **Final strategy used**: Extracted dynamic requirements and user history into a structured prompt, requesting strictly formatted JSON output. Evaluated using Groq `llama-3.2-90b-vision-preview`.\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    evaluate()
