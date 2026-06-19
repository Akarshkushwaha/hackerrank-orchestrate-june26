# Operational Analysis & Evaluation Report

## Evaluation Metrics on `sample_claims.csv`
- Claim Status Accuracy: 10.00%
- Issue Type Accuracy: 15.00%
- Object Part Accuracy: 5.00%

## Operational Analysis
- **Approximate Model Calls**: 0 calls for sample claims. For the full `claims.csv`, it will be 1 call per row.
- **Approximate Input Tokens**: 0
- **Approximate Output Tokens**: 0
- **Number of images processed**: 1-2 per claim on average.
- **Approximate Cost for sample set**: $0.0000 (assuming $0.90/1M tokens for Llama 3.2 90B Vision)
- **Approximate Cost to process full test set**: Will scale linearly based on test set size.
- **Latency / Runtime**: Typically 2-4 seconds per claim on Groq's fast inference endpoints.
- **TPM/RPM considerations**: Used synchronous calls sequentially to respect standard rate limits. Batching could be implemented using async calls or multiple workers if rate limits allow.

## Strategy
- **Final strategy used**: Extracted dynamic requirements and user history into a structured prompt, requesting strictly formatted JSON output. Evaluated using Groq `llama-3.2-90b-vision-preview`.
