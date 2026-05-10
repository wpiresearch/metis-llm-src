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
    Evaluator_user = auto()
    Generalist_System = auto()
    Generalist_User = auto()
    Synthesizer_System = auto()
    Synthesizer_User = auto()


class Prompts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    correctness_prompt: str = """Without citing modern fact-checks, how would you assess this claim on the dimensions of logical consistency, factual accuracy, and contextual appropriateness? 
Consider the contextual appropriateness with the given context. 
Assess each dimension from 0 to 100 and return the response in JSON format {"logical_consistency": "logical consistency", "factual_accuracy": "factual accuracy", "contextual_appropriateness": "contextual appropriateness"}, 
do not include any additional text.
Context: {{original_prompt}}
Claim: {{message}}"""

    experiential_matching_prompt: str = """You are going to measure the matching level of this claim with the given knowledge base and the historical responses respectively.
Consider the given knowledge as the knowledge base, the given history as the historical responses.
Measure the matching level from 0 to 100, which is the lowest to the highest.
Return the response in JSON format {"knowledge_base_matching": "knowledge base matching", "historical_responses_matching": "historical responses matching"}; do not include any additional text.
Knowldege: {{knowledge_base}}
History: {{historical_responses}}
Claim: {{message}}"""

    conflict_information_prompt: str = """You are going to measure the degree of inconsistency and contradictory in the information from the following dimensions: 
a) internal consistency, which measures logical contradictions within the given information
b) disagreement across multiple sources, which compares the given information from multiple sources
c) consistency of information over time, which compares the given information from Temporal Information
Return the response in JSON format {"internal_consistency": "internal consistency", "source_agreement": "source agreement", "temporal_stability": "temporal stability"}; do not include any additional text.
Information: {{message}}
Sources:{{sources}}
Temporal Information: {{temporal_info}}
"""

    problem_importance_prompt: str = """How would you assess the User Prompt for problem importance on the dimensions of potential consequences, temporal urgency, and scope of impact? 
Assess each dimension from 0 to 100 and return the response in JSON format {"potential_consequences": "potential consequences", "temporal_urgency": "temporal urgency", "scope_of_impact": "scope of impact"}, 
do not include any additional text.
User Prompt: {{original_prompt}}"""

    system_two_system_prompt: str = "You are a System Two, logical analytical deep thinking system"

    system_two_user_prompt: str = """Given the previous System One response, and its interpretation of the user's orginal prompt: '{{user_prompt}}', what would you say instead?"""

    domain_expert_system: str = """You are a domain expert, based on prior information from {{previous_node_role}}, provide a more insightful response."""
    domain_expert_user: str = """Based on the conversation thus far, what is your take?"""
    critic_system: str = """You are a critical analyst, challenge assumptions, think logically, previous information is from a {{previous_node_role}}"""
    critic_user: str = """Based on the conversation thus far, what is your assessment?"""
    evaluator_system: str = """You are an evaluator, taking a broader perspective to analyze prior information from {{previous_node_role}} and render an opinion"""
    evaluator_user: str = """Based on the conversation thus far, what is your assessment?"""
    generalist_system: str = """You are a generalist, with a wide base of knowledege, previous information is from {{previous_node_role}}"""
    generalist_user: str = """Based on the conversation thus far, what is your take?"""
    synthesizer_system: str = """You are a synthesizer, you take information from disperate sources and combine it into a concise cogent response, previous information is from {{previous_node_role}}"""
    synthesizer_user: str = """Based on all conversation thus far, what is your synthesis?"""

    jinja_env: Environment = Field(default_factory=Environment, exclude=True)

    def get_prompt(self, prompt: PromptNames, context: dict) -> str:
        prompt_string = getattr(self, prompt.value)
        prompt_template: Template = self.jinja_env.from_string(prompt_string)
        return prompt_template.render(**context)
