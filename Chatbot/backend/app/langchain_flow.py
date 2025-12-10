from openai import OpenAI
import os

class LangChainFlows:
    def __init__(self, openai_api_key, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model_name

    def validate(self, question):
        prompt = f"""
Determine if this is a valid math question. 
Respond ONLY in JSON:

{{
  "is_valid": true/false,
  "reason": "short explanation"
}}

Question: {question}
"""
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return eval(res.choices[0].message.content)

    def refine(self, question, feedback=""):
        prompt = f"""
Refine the following math question. Respond in JSON:

{{
  "revised_question": "...",
  "issues_fixed": ["...", "..."]
}}

Question: {question}

Feedback: {feedback}
"""
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return eval(res.choices[0].message.content)
