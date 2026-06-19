import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def evaluate_justification(ground_truth_justification, predicted_justification):
    """
    LLM-as-a-Judge Evaluation Script
    This function uses an LLM to evaluate the quality of a generated claim justification
    against the ground truth justification provided in sample_claims.csv.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return "API Key not set."
        
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    
    system_prompt = (
        "You are an expert insurance claims judge. "
        "Your task is to compare a predicted justification against a ground truth justification "
        "and score the prediction on a scale of 1 to 5 based on accuracy, relevance, and alignment.\n"
        "Output ONLY the integer score (1-5)."
    )
    
    user_prompt = f"""
    Ground Truth Justification: {ground_truth_justification}
    
    Predicted Justification: {predicted_justification}
    
    Score:"""
    
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage to demonstrate evaluation setup
    print("Setting up LLM-as-a-Judge Evaluation...")
    gt = "The image clearly shows a dent on the rear bumper and the user history does not add risk."
    pred = "Visual evidence confirms rear bumper dent. User history is clear."
    
    score = evaluate_justification(gt, pred)
    print(f"Ground Truth: {gt}")
    print(f"Prediction: {pred}")
    print(f"LLM Judge Score: {score}/5")
