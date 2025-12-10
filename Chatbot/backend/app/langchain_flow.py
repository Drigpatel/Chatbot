import json
from openai import OpenAI


def safe_json_parse(text):
    """
    Safely parse JSON returned by the model.
    Handles cases where GPT returns:
    - Markdown blocks
    - Extra characters
    - Invalid JSON
    """
    if not text or not isinstance(text, str):
        return {"error": "Empty or invalid response", "raw": text}

    # Remove markdown wrappers if present
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {"error": "Invalid JSON response from model", "raw": cleaned}


class LangChainFlows:
    def __init__(self, openai_api_key, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model_name

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------
    def validate(self, question):
        prompt = f"""
Determine if this is a valid math question. 
Respond ONLY in JSON format:

{{
  "is_valid": true,
  "reason": "short explanation"
}}

Question: {question}
"""

        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = res.choices[0].message.content
        return safe_json_parse(raw)

    # -----------------------------------------
    # REFINEMENT
    # -----------------------------------------
    def refine(self, question, feedback=""):
        prompt = f"""
Refine the following math question. Produce ONLY valid JSON. No explanation, no markdown.

Expected Format:
{{
  "revised_question": "string",
  "issues_fixed": ["point1", "point2"]
}}

Question: {question}
Feedback: {feedback}
"""

        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = res.choices[0].message.content
        return safe_json_parse(raw)
