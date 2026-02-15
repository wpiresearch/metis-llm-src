import asyncio
import json
import math
from abc import abstractmethod
from dataclasses import dataclass, fields

import ollama
from nrclex import NRCLex

from prompts import PromptNames, Prompts


@dataclass(unsafe_hash=True)
class ResponseVectors:
    calculated_value: int = 0

    @abstractmethod
    def _compute_value(self) -> int: ...

    def __post_init__(self) -> None:
        self.calculated_value = self._compute_value()


@dataclass(kw_only=True, unsafe_hash=True)
class EmotionalResponse(ResponseVectors):
    version: str = "0.1"

    fear: float
    weight_fear: float = 0.1

    anger: float
    weight_anger: float = 0.1

    anticipation: float
    weight_anticipation: float = 0.1

    trust: float
    weight_trust: float = 0.1

    surprise: float
    weight_surprise: float = 0.1

    positive: float
    weight_positive: float = 0.1

    negative: float
    weight_negative: float = 0.1

    sadness: float
    weight_sadness: float = 0.1

    disgust: float
    weight_disgust: float = 0.1

    joy: float
    weight_joy: float = 0.1

    def _compute_value(self) -> int:
        self_fields = fields(self)
        running_total = 0.0
        for field in self_fields:
            if (
                field.name != "version"
                and field.name != "calculated_value"
                and not field.name.startswith("weight_")
            ):
                running_total += getattr(self, field.name) * getattr(
                    self, f"weight_{field.name}"
                )
        return min(int(running_total), 100)


@dataclass(kw_only=True, unsafe_hash=True)
class CorrectnessResponse(ResponseVectors):
    version: str = "0.1"

    # Sum of weights should be 1
    logical_consistency: float
    weight_logical_consistency: float = 0.3

    factual_accuracy: float
    weight_factual_accuracy: float = 0.4

    contextual_appropriateness: float
    weight_contextual_appropriateness: float = 0.3

    def _compute_value(self) -> int:
        return min(
            int(
                (self.logical_consistency * self.weight_logical_consistency)
                + (self.factual_accuracy * self.weight_factual_accuracy)
                + (
                    self.contextual_appropriateness
                    * self.weight_contextual_appropriateness
                )
            ),
            100,
        )


@dataclass(kw_only=True, unsafe_hash=True)
class ExperientialMatchingResponse(ResponseVectors):
    version: str = "0.1"

    # Weights are adaptive based on context. Should there be constraints on weights?
    knowledge_base_matching: float
    weight_knowledge_base_matching: float = 0.5

    historical_responses_matching: float
    weight_historical_responses_matching: float = 0.5

    def _compute_value(self) -> int:
        # This calculation assumes the matching values are in the range of [0,100]
        return min(
            int(
                (self.knowledge_base_matching * self.weight_knowledge_base_matching)
                + (
                    self.historical_responses_matching
                    * self.weight_historical_responses_matching
                )
            ),
            100,
        )


@dataclass(kw_only=True, unsafe_hash=True)
class ConflictInformation(ResponseVectors):
    version: str = "0.1"

    # Sum of weights should be 1
    internal_consistency: float
    weight_internal_consistency: float = 0.3

    source_agreement: float
    weight_source_agreement: float = 0.4

    temporal_stability: float
    weight_temporal_stability: float = 0.3

    def _compute_value(self) -> int:
        return min(
            int(
                (self.internal_consistency * self.weight_internal_consistency)
                + (self.source_agreement * self.weight_source_agreement)
                + (self.temporal_stability * self.weight_temporal_stability)
            ),
            100,
        )


@dataclass(kw_only=True, unsafe_hash=True)
class ProblemImportance(ResponseVectors):
    version: str = "0.1"

    potential_consequences: float
    weight_potential_consequences: float = 0.4

    temporal_urgency: float
    weight_temporal_urgency: float = 0.3

    scope_of_impact: float
    weight_scope_of_impact: float = 0.3

    def _compute_value(self) -> int:
        return min(
            int(
                (self.potential_consequences * self.weight_potential_consequences)
                + (self.temporal_urgency * self.weight_temporal_urgency)
                + (self.scope_of_impact * self.weight_scope_of_impact)
            ),
            100,
        )


@dataclass(kw_only=True, unsafe_hash=True)
class MetacognitiveVector(ResponseVectors):
    version: str = "0.1"
    emotional_response: EmotionalResponse
    weight_emotional_response: float = 0.2

    correctness: CorrectnessResponse
    weight_correctness: float = 0.2

    experiential_matching: ExperientialMatchingResponse
    weight_experiential_matching: float = 0.2

    conflict_information: ConflictInformation
    weight_conflict_information: float = 0.2

    problem_importance: ProblemImportance
    weight_problem_importance: float = 0.2

    activation_threshold: float = 0.1

    def should_engage_system_two(self) -> bool:
        activation_value = self._activation_function(self.calculated_value)
        return activation_value >= self.activation_threshold

    def _activation_function(self, value: int) -> float:
        return 1 / (1 + math.exp(-value * 0.00001))

    def _compute_value(self) -> int:
        return int(
            (self.emotional_response._compute_value() * self.weight_emotional_response)
            + (self.correctness._compute_value() * self.weight_correctness)
            + (
                self.experiential_matching._compute_value()
                * self.weight_experiential_matching
            )
            + (
                self.conflict_information._compute_value()
                * self.weight_conflict_information
            )
            + (
                self.problem_importance._compute_value()
                * self.weight_problem_importance
            )
        )


async def compute_metacognitive_state_vector(
    prompts: Prompts,
    weights: dict[str, dict[str, float]],
    response: str,
    original_prompt: str,
    knowledge_base: str = "",
    historical_responses: str = "",
    sources: str = "",
    temporal_info: str = "",
) -> MetacognitiveVector:
    prompts = Prompts()
    (
        emotional_response,
        correctness,
        experiential_matching,
        conflict_information,
        problem_importance,
    ) = await asyncio.gather(
        _compute_emotional_response(response, weights["emotional_response"]),
        _compute_correctness(
            response, original_prompt, prompts, weights["correctness"]
        ),
        _compute_experiential_matching(
            response,
            knowledge_base,
            historical_responses,
            prompts,
            weights["experiential_matching"],
        ),
        _compute_conflict_information(
            response, sources, temporal_info, prompts, weights["conflict_information"]
        ),
        _compute_problem_importance(response, prompts, weights["problem_importance"]),
    )

    return MetacognitiveVector(
        emotional_response=emotional_response,
        correctness=correctness,
        experiential_matching=experiential_matching,
        conflict_information=conflict_information,
        problem_importance=problem_importance,
        **weights["msv_weights"],
    )


async def _compute_emotional_response(
    message: str, weights: dict[str, float]
) -> EmotionalResponse:
    text_object = NRCLex(message)
    # remove vestigial(?) "anticip" in favor of the populated "anticipation",
    # seems like sometimes "anticipation" is populated sometimes "anticip" ?
    if "anticipation" not in text_object.affect_frequencies:
        if "anticip" in text_object.affect_frequencies:
            text_object.affect_frequencies["anticipation"] = (
                text_object.affect_frequencies["anticip"]
            )
        else:
            text_object.affect_frequencies["anticipation"] = 0.0
    del text_object.affect_frequencies["anticip"]
    return EmotionalResponse(
        **{k: v * 100 for k, v in text_object.affect_frequencies.items()}
    )


async def _compute_correctness(
    message: str, original_prompt: str, prompts: Prompts, weights: dict[str, float]
) -> CorrectnessResponse:
    content = prompts.get_prompt(
        PromptNames.Correctness,
        {"original_prompt": original_prompt, "message": message},
    )
    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return CorrectnessResponse(
            logical_consistency=parsed_response["logical_consistency"],
            factual_accuracy=int(parsed_response["factual_accuracy"]),
            contextual_appropriateness=int(
                parsed_response["contextual_appropriateness"]
            ),
            **weights,
        )
    except:
        return CorrectnessResponse(
            logical_consistency=0.0,
            factual_accuracy=0.0,
            contextual_appropriateness=0.0,
            **weights,
        )


# Depending how to input knowledge base and historical responses, the prompt template would be different.
# How to prompt to get matching level? options: matching level [0,100], similarity [0,1]
async def _compute_experiential_matching(
    message: str,
    knowledge_base: str,
    historical_responses: str,
    prompts: Prompts,
    weights: dict[str, float],
) -> ExperientialMatchingResponse:
    content = prompts.get_prompt(
        PromptNames.Experiential_Matching,
        {
            "knowledge_base": knowledge_base,
            "message": message,
            "historical_responses": historical_responses,
        },
    )
    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return ExperientialMatchingResponse(
            knowledge_base_matching=float(parsed_response["knowledge_base_matching"]),
            historical_responses_matching=float(
                parsed_response["historical_responses_matching"]
            ),
            **weights,
        )
    except:
        return ExperientialMatchingResponse(
            knowledge_base_matching=0.0, historical_responses_matching=0.0, **weights
        )


async def _compute_conflict_information(
    message: str,
    sources: str,
    temporal_info: str,
    prompts: Prompts,
    weights: dict[str, float],
) -> ConflictInformation:
    content = prompts.get_prompt(
        PromptNames.Conflict_Information,
        {"sources": sources, "message": message, "temporal_info": temporal_info},
    )
    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return ConflictInformation(
            internal_consistency=float(parsed_response["internal_consistency"]),
            source_agreement=float(parsed_response["source_agreement"]),
            temporal_stability=float(parsed_response["temporal_stability"]),
            **weights,
        )
    except:
        return ConflictInformation(
            internal_consistency=0.0,
            source_agreement=0.0,
            temporal_stability=0.0,
            **weights,
        )


async def _compute_problem_importance(
    original_prompt: str, prompts: Prompts, weights: dict[str, float]
) -> ProblemImportance:
    content = prompts.get_prompt(
        PromptNames.Problem_Importance, {"original_prompt": original_prompt}
    )
    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return ProblemImportance(
            potential_consequences=float(parsed_response["potential_consequences"]),
            temporal_urgency=float(parsed_response["temporal_urgency"]),
            scope_of_impact=float(parsed_response["scope_of_impact"]),
            **weights,
        )
    except:
        return ProblemImportance(
            potential_consequences=0.0,
            temporal_urgency=0.0,
            scope_of_impact=0.0,
            **weights,
        )


def generate_empty_msv() -> MetacognitiveVector:
    emotional_response = EmotionalResponse(
        fear=0,
        anger=0,
        anticipation=0,
        trust=0,
        surprise=0,
        positive=0,
        negative=0,
        sadness=0,
        disgust=0,
        joy=0,
    )
    correctness = CorrectnessResponse(
        logical_consistency=0.0, factual_accuracy=0.0, contextual_appropriateness=0.0
    )
    experiential_matching = ExperientialMatchingResponse(
        knowledge_base_matching=0.0, historical_responses_matching=0.0
    )
    conflict_information = ConflictInformation(
        internal_consistency=0.0, source_agreement=0.0, temporal_stability=0.0
    )
    problem_importance = ProblemImportance(
        potential_consequences=0.0, temporal_urgency=0.0, scope_of_impact=0.0
    )

    return MetacognitiveVector(
        emotional_response=emotional_response,
        correctness=correctness,
        experiential_matching=experiential_matching,
        conflict_information=conflict_information,
        problem_importance=problem_importance,
    )
