import os
import json
import time
from groq import Groq
from pathlib import Path
from utils import encode_image_base64, get_user_history, get_evidence_requirements
from prompts import SYSTEM_PROMPT, build_user_prompt

class ClaimsProcessor:
    def __init__(self, api_key=None, model="llama-3.2-90b-vision-instruct"):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY must be set in environment variables or passed explicitly.")
        self.client = Groq(api_key=key)
        self.model = model
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def process_claim(self, row, history_df, req_df, base_dir="."):
        user_id = row['user_id']
        image_paths = row['image_paths'].split(';')
        user_claim = row['user_claim']
        claim_object = row['claim_object']
        
        # Get context
        user_history = get_user_history(user_id, history_df)
        evidence_reqs = get_evidence_requirements(claim_object, req_df)
        
        image_ids = [Path(p).stem for p in image_paths]
        
        # Build prompt
        user_prompt_text = build_user_prompt(claim_object, user_claim, user_history, evidence_reqs, image_ids)
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text}
                ]
            }
        ]
        
        # Attach images
        for img_path in image_paths:
            full_path = os.path.join(base_dir, "dataset", img_path)
            if not os.path.exists(full_path):
                print(f"Warning: image not found at {full_path}")
                continue
            
            ext = Path(full_path).suffix.lower().replace('.', '')
            mime_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else f"image/{ext}"
            
            b64_image = encode_image_base64(full_path)
            if b64_image:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}"
                    }
                })
        
        # Call Groq API
        try:
            # We enforce JSON output using response_format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            self.total_calls += 1
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            
            result_text = response.choices[0].message.content
            try:
                result_json = json.loads(result_text)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {user_id}. Raw: {result_text}")
                result_json = self._get_fallback_json()
                
            # Add original inputs
            result_json['user_id'] = user_id
            result_json['image_paths'] = row['image_paths']
            result_json['user_claim'] = user_claim
            result_json['claim_object'] = claim_object
            return result_json
            
        except Exception as e:
            print(f"Error processing claim {user_id}: {e}")
            fallback = self._get_fallback_json()
            fallback['user_id'] = user_id
            fallback['image_paths'] = row['image_paths']
            fallback['user_claim'] = user_claim
            fallback['claim_object'] = claim_object
            return fallback

    def _get_fallback_json(self):
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "API Error or JSON parsing failed",
            "risk_flags": "manual_review_required",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "System encountered an error processing the images.",
            "supporting_image_ids": "none",
            "valid_image": False,
            "severity": "unknown"
        }
