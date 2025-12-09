import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


VALIDATION_PROMPT = """
You are a validator. Determine if the user message is a proper mathematical question.
Return a VALID JSON object only, with:
- is_valid: true or false
- reason: explanation only if invalid

User Message:
{question}
Output only valid JSON:
"""

REFINEMENT_PROMPT = """
Rewrite the question clearly and correctly using proper math formatting.

Return a VALID JSON object with:
- revised_question: improved question as a string
- issues_fixed: bullet list of improvements

Original Question:
{question}

User feedback:
{feedback}

Output only valid JSON:
"""


class LangChainFlows:
    def __init__(self, openai_api_key, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.3  # Better consistency for JSON
        )

        self.validation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=VALIDATION_PROMPT,
                input_variables=["question"]
            )
        )

        self.refinement_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=REFINEMENT_PROMPT,
                input_variables=["question", "feedback"]
            )
        )

    def _safe_parse(self, raw: str):
        try:
            return json.loads(raw)
        except Exception:
            return {
                "error": "JSON Parse Failed",
                "raw_output": raw
            }

    def validate(self, question: str):
        raw = self.validation_chain.run(question=question)
        return self._safe_parse(raw)

    def refine(self, question: str, feedback: str = ""):
        raw = self.refinement_chain.run(question=question, feedback=feedback)
        return self._safe_parse(raw)
