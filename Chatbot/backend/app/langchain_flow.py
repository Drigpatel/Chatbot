import json
from openai import OpenAI


def safe_json_parse(text):
    """
    Safely parse JSON returned by GPT.
    Ensures NO crashing even if output is garbage.
    """
    if not text or not isinstance(text, str):
        return {"error": "Empty response", "raw": text}

    cleaned = (
        text.replace("```json", "")
            .replace("```", "")
            .strip()
    )

    # Fix common GPT mistakes: single quotes, trailing commas
    cleaned = cleaned.replace("'", "\"")

    try:
        return json.loads(cleaned)
    except Exception:
        # Return raw text so backend does not crash
        return {"error": "Invalid JSON", "raw": cleaned}


class LangChainFlows:
    def __init__(self, openai_api_key, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model_name

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------
    def validate(self, question):
        prompt = f"""
Respond ONLY with JSON:

{{
  "is_valid": true or false,
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
Refine the question. Respond ONLY in this JSON:

{{
  "revised_question": "text",
  "issues_fixed": ["a", "b"]
}}

Original: {question}
Feedback: {feedback}
"""

        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = res.choices[0].message.content
        return safe_json_parse(raw)
