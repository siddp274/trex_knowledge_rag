"""
Query Step 4: Generate the final answer with GPT-4o.

Takes the assembled context (passages) and the user's question,
produces a grounded answer with passage citations.

This is the only step that uses the expensive query model (GPT-4o).
Everything else in the query pipeline uses gpt-4.1-mini or no LLM at all.
One GPT-4o call per user query — that's the cost model.
"""
import logging
from openai import OpenAI

from config import TREXConfig

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """You are an expert research assistant with access to a structured knowledge base.

You will be given retrieved document passages — some are raw text chunks (Level 0), 
some are higher-level thematic summaries from a RAPTOR tree (Level 1-2).

Your task: answer the user's question accurately and completely.

Rules:
- Draw on ALL provided passages
- Cite specific passages where relevant: [Passage 1], [Passage 3]
- If the context is insufficient to fully answer, say so clearly
- Do not make up facts not supported by the context
- Write in clear, professional prose

{context_data}"""


class AnswerGenerator:
    """Generate grounded answers using GPT-4o."""

    def __init__(self, config: TREXConfig):
        self.config = config
        self.client = OpenAI(base_url=config.openai_query_endpoint, api_key=config.openai_query_api_key)

    def _build_context_data(self, assembled_context: dict) -> str:
        """Format the passages section for the system prompt."""
        passages_text = assembled_context.get("passages_text", "")
        if passages_text:
            return f"## Retrieved Passages\n\n{passages_text}"
        return ""

    def generate(self, question: str, assembled_context: dict) -> dict:
        """
        Generate an answer grounded in the retrieved context.
        
        Returns:
            {
                "answer": str,
                "model": str,
                "context_tokens": int,
                "passages_used": int,
                "prompt_tokens": int,
                "completion_tokens": int,
            }
        """
        system_prompt = ANSWER_SYSTEM_PROMPT.format(
            context_data=self._build_context_data(assembled_context)
        )

        response = self.client.chat.completions.create(
            model=self.config.query_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        logger.info(
            f"[Answer] Generated ({usage.prompt_tokens} prompt + "
            f"{usage.completion_tokens} completion tokens)"
        )

        return {
            "answer": answer,
            "model": self.config.query_model,
            "context_tokens": assembled_context.get("total_tokens", 0),
            "passages_used": assembled_context.get("num_passages", 0),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }

    def generate_stream(self, question: str, assembled_context: dict):
        """
        Streaming version — yields answer tokens as they arrive.
        Use this for real-time UX.
        """
        system_prompt = ANSWER_SYSTEM_PROMPT.format(
            context_data=self._build_context_data(assembled_context)
        )

        stream = self.client.chat.completions.create(
            model=self.config.query_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
