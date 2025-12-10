import json
from openai import OpenAI


def safe_json_parse(text):
    """
    Safely parse JSON returned from GPT.
    Ensures NO crashes even when GPT returns invalid JSON.
    """
    if not text or not isinstance(text, str):
        return {"error": "Empty GPT response", "raw": text}

    cleaned = (
        text.replace("```json", "")
            .replace("```", "")
            .strip()
    )

    # Convert single quotes → double quotes
    cleaned = cleaned.replace("'", "\"")

    # Try normal JSON load
    try:
        return json.loads(cleaned)
    except Exception:
        # Last fallback → still return usable structure
        return {"error": "Invalid JSON", "raw": cleaned}


class LangChainFlows:
    def __init__(self, openai_api_key, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model_name

    # ---------------------------
    # VALIDATE
    # ---------------------------
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
        print("RAW VALIDATE OUTPUT:", raw)

        return safe_json_parse(raw)

    # ---------------------------
    # REFINE
    # ---------------------------
    def refine(self, question, feedback=""):
        prompt = f"""
Refine the question. Respond ONLY in JSON:

{{
  "revised_question": "text",
  "issues_fixed": ["item1", "item2"]
}}

Original: {question}
Feedback: {feedback}
"""

        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = res.choices[0].message.content
        print("RAW REFINE OUTPUT:", raw)

        return safe_json_parse(raw)
