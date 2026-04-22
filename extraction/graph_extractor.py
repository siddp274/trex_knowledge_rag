"""
Step 3a: Extract entities and relationships from each TextUnit.

Adapted from GraphRAG's GraphExtractor class. Key changes:
- Uses OpenAI client directly (not their internal LLM wrapper)
- Synchronous (their version is async — we'll keep it simple)
- Parses the same delimiter-based format: <|> for fields, ## for records
- Supports gleaning (re-extraction passes to catch missed entities)

The output is raw — deduplication happens in Step 3b.
"""
import logging
import re
from typing import Any
from openai import OpenAI

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from ingestion.text_unit import TextUnit
from prompts.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    CONTINUE_PROMPT,
    LOOP_PROMPT,
)

logger = logging.getLogger(__name__)

# Delimiters — must match the prompt
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# Entity types we extract. Customize per domain.
DEFAULT_ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY",
    "CONCEPT", "EVENT", "PRODUCT", "METRIC",
]


class GraphExtractor:
    """
    Extracts entities and relationships from text using an LLM.
    
    Follows GraphRAG's pattern:
    1. Send the text + prompt → get initial extraction
    2. (Gleaning) Ask "did you miss anything?" → get more entities
    3. Parse the delimiter-based output into structured records
    """

    def __init__(
        self,
        config: TREXConfig,
        entity_types: list[str] | None = None,
        max_gleanings: int = 1,
    ):
        self.config = config
        self.client = OpenAI(
            base_url = config.openai_extractor_endpoint,
            api_key = config.openai_extractor_api_key,
        )
        self.model = config.extraction_model
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.max_gleanings = max_gleanings

    def extract_from_text_unit(
        self,
        text_unit: TextUnit,
    ) -> dict[str, Any]:
        """
        Extract entities and relationships from a single TextUnit.
        
        Returns:
            {
                "source_id": str,          # TextUnit ID
                "entities": [{"title": str, "type": str, "description": str}],
                "relationships": [{"source": str, "target": str, 
                                   "description": str, "weight": float}]
            }
        """
        try:
            raw_output = self._run_extraction(text_unit.text)
            entities, relationships = self._parse_output(raw_output)

            return {
                "source_id": text_unit.id,
                "entities": entities,
                "relationships": relationships,
            }

        except Exception as e:
            logger.error(
                f"[Extract] Failed on TextUnit {text_unit.id[:12]}...: {e}"
            )
            return {
                "source_id": text_unit.id,
                "entities": [],
                "relationships": [],
            }

    def _run_extraction(self, text: str) -> str:
        """
        Run the extraction prompt + gleaning loop.
        
        GraphRAG's gleaning approach:
        1. Initial extraction
        2. For each gleaning pass:
           a. Send CONTINUE_PROMPT → get more entities
           b. Send LOOP_PROMPT → ask if there are still more
           c. If "N", stop. If "Y", continue to next pass.
        """
        # Build the initial prompt
        prompt_text = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=",".join(self.entity_types),
            input_text=text,
        )

        messages = [{"role": "user", "content": prompt_text}]

        # Initial extraction
        response = self._call_llm(messages)
        results = response
        messages.append({"role": "assistant", "content": response})

        # Gleaning passes
        for i in range(self.max_gleanings):
            # Ask for missed entities
            messages.append({"role": "user", "content": CONTINUE_PROMPT})
            response = self._call_llm(messages)
            messages.append({"role": "assistant", "content": response})
            results += response

            # Don't bother with loop check on last gleaning
            if i >= self.max_gleanings - 1:
                break

            # Ask if there are more
            messages.append({"role": "user", "content": LOOP_PROMPT})
            response = self._call_llm(messages)
            messages.append({"role": "assistant", "content": response})

            if response.strip().upper() != "Y":
                break

        return results

    def _call_llm(self, messages: list[dict]) -> str:
        """Call the LLM with retry on failure."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _parse_output(
        self,
        raw_output: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        Parse the delimiter-based output into structured records.
        
        GraphRAG format:
          ("entity"<|>ENTITY_NAME<|>ENTITY_TYPE<|>DESCRIPTION)
          ##
          ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>WEIGHT)
          ##
          <|COMPLETE|>
        """
        entities = []
        relationships = []

        records = [r.strip() for r in raw_output.split(RECORD_DELIMITER)]

        for raw_record in records:
            # Strip parentheses
            record = re.sub(r"^\(|\)$", "", raw_record.strip())
            if not record or record == COMPLETION_DELIMITER:
                continue

            fields = record.split(TUPLE_DELIMITER)
            record_type = fields[0].strip().strip('"').lower()

            if record_type == "entity" and len(fields) >= 4:
                entity_name = self._clean(fields[1]).upper()
                entity_type = self._clean(fields[2]).upper()
                entity_desc = self._clean(fields[3])

                if entity_name:  # skip empty names
                    entities.append({
                        "title": entity_name,
                        "type": entity_type,
                        "description": entity_desc,
                    })

            elif record_type == "relationship" and len(fields) >= 5:
                source = self._clean(fields[1]).upper()
                target = self._clean(fields[2]).upper()
                description = self._clean(fields[3])

                try:
                    weight = float(fields[-1].strip().rstrip(")"))
                except ValueError:
                    weight = 1.0

                if source and target:  # skip if either is empty
                    relationships.append({
                        "source": source,
                        "target": target,
                        "description": description,
                        "weight": weight,
                    })

        return entities, relationships

    @staticmethod
    def _clean(value: str) -> str:
        """Strip whitespace and surrounding quotes."""
        return value.strip().strip('"').strip("'").strip()