"""
Prompts for entity/relationship extraction.
Adapted from GraphRAG's extract_graph.py prompts.

Key design choices:
- Delimiter-based output (<|> and ##) instead of JSON.
  GraphRAG found this dramatically more reliable — LLMs hallucinate 
  JSON structure (missing brackets, trailing commas) but rarely 
  mess up simple delimiters.
- Entity names are UPPERCASED in the prompt examples.
  This is GraphRAG's normalization-at-source trick — the LLM does 
  the uppercasing for us, so "openai", "OpenAI", "OPENAI" all 
  come out as "OPENAI".
- Gleaning prompts ask the LLM to re-check for missed entities.
  With 1200-token chunks, one pass catches ~85% of entities;
  one glean pass brings it to ~95%.
"""

ENTITY_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, UPPERCASED
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.

######################
Output:
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

# After first extraction, ask the LLM to look again for missed entities.
# GraphRAG calls this "gleaning."
CONTINUE_PROMPT = (
    "MANY entities and relationships were missed in the last extraction. "
    "Remember to ONLY emit entities that match any of the previously extracted types. "
    "Add them below using the same format:\n"
)

# Ask whether more gleaning passes are needed.
LOOP_PROMPT = (
    "It appears some entities and relationships may have still been missed. "
    "Answer Y if there are still entities or relationships that need to be added, "
    "or N if there are none. Please answer with a single letter Y or N.\n"
)

# Used in Step 3c to consolidate duplicate descriptions.
DESCRIPTION_SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary 
of the data provided below. Given one or two entities, and a list of descriptions, 
all related to the same entity or group of entities.

Please concatenate all of these into a single, comprehensive description. 
Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions 
and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we 
have full context.

#######
---Descriptions---
{description_list}
#######
Output:
"""