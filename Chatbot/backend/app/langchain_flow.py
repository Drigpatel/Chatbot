import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

VALIDATION_PROMPT = """
You are a validator. Determine if the user message is a proper mathematical question.
Return a VALID JSON object only with:
- is_valid: true or false
- reason: explanation only if invalid

User Message:
{question}

Output JSON only:
"""

REFINEMENT_PROMPT = """
Rewrite the question clearly and correctly using proper mathematical wording.

Return a VALID JSON object only with:
- revised_question: improved question
- issues_fixed: bullet list of improvements

Original Question:
{question}

User Feedback:
{feedback}

Output JSON only:
"""

class LangChainFlows:
    def __init__(self, openai_api_key, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1
        )

        self.validation_prompt = PromptTemplate(
            template=VALIDATION_PROMPT,
            input_variables=["question"]
        )

        self.refinement_prompt = PromptTemplate(
            template=REFINEMENT_PROMPT,
            input_variables=["question", "feedback"]
        )

    def _parse_json(self, text: str):
        """Safely extract JSON from an LLM response."""
        text = text.strip()
        try:
            # Attempt direct JSON parsing
            return json.loads(text)
        except:
            pass

        # Try extracting JSON inside text
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except:
            return {"revised_question": text, "issues_fixed": ["Model returned non-JSON output"]}

    def validate(self, question: str):
        result = self.llm.invoke(
            self.validation_prompt.format(question=question)
        )
        return self._parse_json(result.content)

    def refine(self, question: str, feedback: str = ""):
        result = self.llm.invoke(
            self.refinement_prompt.format(question=question, feedback=feedback or "")
        )
        return self._parse_json(result.content)
