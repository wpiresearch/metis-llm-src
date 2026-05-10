# =============================================================================
# CHANGELOG — prompts.py
# Last modified: 2026-02-14 (RJS)
#
# Changes from previous version:
#
# 1. BUG FIX (conflict_information_prompt):
#    Prompt never specified a numeric 0-100 range. LLMs returned qualitative
#    strings like "High", "low", "inconsistent" instead of numbers, causing
#    every CI computation to fall through to the zero-fallback. Added
#    explicit "Assess each dimension from 0 to 100" instruction and numeric
#    JSON example, matching the pattern of all other prompts.
#
# 2. BUG FIX (all prompts — JSON examples used string placeholders):
#    All prompts showed JSON examples with string values like
#    {"logical_consistency": "logical consistency"} which encouraged LLMs
#    to return strings instead of numbers. Changed all examples to use
#    numeric values like {"logical_consistency": 75}. This was the root
#    cause of the "can't multiply sequence by non-int of type 'float'"
#    errors in correctness parsing.
#
# 3. IMPROVEMENT (all prompts — stronger JSON-only instruction):
#    Added "Return ONLY valid JSON with numeric values" to reinforce that
#    no prose, no markdown fences, and no string values should be returned.
#
# 4. BUG FIX (Evaluator_user casing):
#    Was `Evaluator_user` (lowercase 'u') — should be `Evaluator_User`
#    for consistency with all other role prompts. This caused a mismatch
#    when role_assignment.py constructed PromptNames via f-string:
#    `PromptNames(f"{role}_user")` would not match `evaluator_user` since
#    auto() generates from the attribute name.
# =============================================================================

from enum import StrEnum, auto

from jinja2 import Environment, Template
from pydantic import BaseModel, ConfigDict, Field


class PromptNames(StrEnum):
    Correctness = "correctness_prompt"
    Experiential_Matching = "experiential_matching_prompt"
    Conflict_Information = "conflict_information_prompt"
    Problem_Importance = "problem_importance_prompt"
    System_Two_System = "system_two_system_prompt"
    System_Two_User = "system_two_user_prompt"
    Domain_Expert_System = auto()
    Domain_Expert_User = auto()
    Critic_System = auto()
    Critic_User = auto()
    Evaluator_System = auto()
    Evaluator_User = auto()
    Generalist_System = auto()
    Generalist_User = auto()
    Synthesizer_System = auto()
    Synthesizer_User = auto()


class Prompts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    correctness_prompt: str = """Without citing modern fact-checks, how would you assess this claim on the dimensions of logical consistency, factual accuracy, and contextual appropriateness? 
Consider the contextual appropriateness with the given context. 
Assess each dimension from 0 to 100 and return ONLY valid JSON with numeric values, no additional text.
Example format: {"logical_consistency": 75, "factual_accuracy": 80, "contextual_appropriateness": 65}
Context: {{original_prompt}}
Claim: {{message}}"""

    experiential_matching_prompt: str = """You are going to measure the matching level of this claim with the given knowledge base and the historical responses respectively.
Consider the given knowledge as the knowledge base, the given history as the historical responses.
Measure the matching level from 0 to 100, where 0 is the lowest and 100 is the highest.
Return ONLY valid JSON with numeric values, no additional text.
Example format: {"knowledge_base_matching": 60, "historical_responses_matching": 45}
Knowledge: {{knowledge_base}}
History: {{historical_responses}}
Claim: {{message}}"""

    conflict_information_prompt: str = """You are going to measure the degree of inconsistency and contradiction in the information from the following dimensions: 
a) internal consistency, which measures logical contradictions within the given information
b) source agreement, which compares the given information across multiple sources
c) temporal stability, which compares the given information against the Temporal Information
Assess each dimension from 0 to 100, where 0 means no conflict and 100 means extreme conflict.
Return ONLY valid JSON with numeric values, no additional text.
Example format: {"internal_consistency": 30, "source_agreement": 55, "temporal_stability": 40}
Information: {{message}}
Sources: {{sources}}
Temporal Information: {{temporal_info}}
"""

    problem_importance_prompt: str = """How would you assess the User Prompt for problem importance on the dimensions of potential consequences, temporal urgency, and scope of impact? 
Assess each dimension from 0 to 100 and return ONLY valid JSON with numeric values, no additional text.
Example format: {"potential_consequences": 70, "temporal_urgency": 45, "scope_of_impact": 60}
User Prompt: {{original_prompt}}"""

    system_two_system_prompt: str = "You are a System Two, logical analytical deep thinking system"

    system_two_user_prompt: str = """Given the previous System One response, and its interpretation of the user's orginal prompt: '{{user_prompt}}', what would you say instead?"""

    domain_expert_system: str = """You are a domain expert, based on prior information from {{previous_node_role}}, provide a more insightful response."""
    domain_expert_user: str = """The user's original question is: '{{user_prompt}}'. Based on the conversation thus far, what is your expert take?"""
    critic_system: str = """You are a critical analyst, challenge assumptions, think logically, previous information is from a {{previous_node_role}}"""
    critic_user: str = """The user's original question is: '{{user_prompt}}'. Based on the conversation thus far, what is your critical assessment?"""
    evaluator_system: str = """You are an evaluator, taking a broader perspective to analyze prior information from {{previous_node_role}} and render an opinion"""
    evaluator_user: str = """The user's original question is: '{{user_prompt}}'. Based on the conversation thus far, what is your evaluation?"""
    generalist_system: str = """You are a generalist, with a wide base of knowledege, previous information is from {{previous_node_role}}"""
    generalist_user: str = """The user's original question is: '{{user_prompt}}'. Based on the conversation thus far, what is your take as a generalist?"""
    synthesizer_system: str = """You are a synthesizer, you take information from disperate sources and combine it into a concise cogent response, previous information is from {{previous_node_role}}"""
    synthesizer_user: str = """Based on all conversation thus far, what is your synthesis?"""

    jinja_env: Environment = Field(default_factory=Environment, exclude=True)

    def get_prompt(self, prompt: PromptNames, context: dict) -> str:
        prompt_string = getattr(self, prompt.value)
        prompt_template: Template = self.jinja_env.from_string(prompt_string)
        return prompt_template.render(**context)
