# =============================================================================
# CHANGELOG — metacognitive.py
# Last modified: 2026-02-14 (RJS)
#
# Changes from previous version:
#
# 1. BUG FIX (compute_metacognitive_state_vector):
#    `prompts = Prompts()` on the first line of the function was shadowing
#    the `prompts` parameter passed in by the caller. The caller's Prompts
#    instance was silently ignored and a fresh default was always constructed.
#    Removed the shadowing assignment so the caller's prompts are used.
#
# 2. BUG FIX (compute_metacognitive_state_vector → _compute_problem_importance call):
#    Was passing `response` (the LLM's answer) as the first positional arg,
#    but _compute_problem_importance's first parameter is `original_prompt`
#    (the user's query). Problem Importance should assess the *query*, not
#    the response. Changed to pass `original_prompt`.
#
# 3. BUG FIX (_compute_emotional_response, "anticip" deletion):
#    `del text_object.affect_frequencies["anticip"]` was unconditional but
#    "anticip" is not always present in NRCLex output, causing KeyError.
#    Changed to `pop("anticip", None)` for safe removal.
#
# 4. BUG FIX (_compute_emotional_response, weights ignored):
#    The `weights` parameter was accepted but never passed to the
#    EmotionalResponse constructor. Config-specified ER dimension weights
#    were silently dropped and class defaults (all 0.1) were always used.
#    Now passes `**weights` to the constructor, consistent with all other
#    _compute_* functions.
#
# 5. BUG FIX (_compute_emotional_response, ER always the same value):
#    NRCLex affect_frequencies are proportions that always sum to 1.0.
#    With equal weights (all 0.1 × 10 emotions), the weighted sum is
#    always the constant ~10 regardless of text content. Individual
#    emotions varied but the aggregate calculated_value was meaningless.
#    FIX: Now uses raw emotion word counts normalized by total word count,
#    so each emotion dimension is independent (not constrained to sum to
#    a constant). Texts with more emotional content score higher overall.
#
# 6. IMPROVEMENT (bare except clauses):
#    Changed bare `except:` to `except Exception:` in all _compute_*
#    functions so KeyboardInterrupt and SystemExit are not swallowed.
#    Added logging of caught exceptions for easier debugging.
#
# 7. IMPROVEMENT (robust numeric parsing):
#    Added _safe_float() helper. All parsed LLM response values now go
#    through float() conversion to handle cases where LLMs return string
#    numbers like "85" instead of bare 85. Also catches the case where
#    _compute_correctness used raw dict values for logical_consistency
#    without float() conversion, causing "can't multiply sequence by
#    non-int of type 'float'" errors.
#
# 8. EXISTING (from prior update, documented here for completeness):
#    All _compute_* functions and compute_metacognitive_state_vector accept
#    an optional `model` parameter (default "llama3.2") so each agent can
#    self-assess using its own model. All existing callers are unaffected
#    (backward-compatible default).
#
# NOTE: These fixes work in tandem with prompts.py changes (same date)
#    that add numeric JSON examples and explicit 0-100 ranges to all
#    prompts (especially conflict_information_prompt which had no numeric
#    range specified at all).
# =============================================================================

import asyncio
import json
import logging
import math
from abc import abstractmethod
from dataclasses import dataclass, fields

import ollama
from nrclex import NRCLex

from prompts import PromptNames, Prompts

logger = logging.getLogger(__name__)


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, returning default if conversion fails.
    Handles LLM responses that return "85" (string) instead of 85 (number).
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


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
    model: str = "llama3.2",
) -> MetacognitiveVector:
    # FIX #1: Removed `prompts = Prompts()` which was shadowing the parameter.
    # The caller's Prompts instance is now actually used.
    (
        emotional_response,
        correctness,
        experiential_matching,
        conflict_information,
        problem_importance,
    ) = await asyncio.gather(
        _compute_emotional_response(response, weights["emotional_response"]),
        _compute_correctness(
            response, original_prompt, prompts, weights["correctness"],
            model=model,
        ),
        _compute_experiential_matching(
            response,
            knowledge_base,
            historical_responses,
            prompts,
            weights["experiential_matching"],
            model=model,
        ),
        _compute_conflict_information(
            response, sources, temporal_info, prompts, weights["conflict_information"],
            model=model,
        ),
        # FIX #2: Was `response` here — should be `original_prompt`.
        # Problem Importance assesses the *query*, not the LLM's answer.
        _compute_problem_importance(original_prompt, prompts, weights["problem_importance"],
            model=model,
        ),
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

    # FIX #5: Previous code used affect_frequencies (proportions summing to 1.0).
    # With equal weights (all 0.1 × 10 emotions), the weighted sum was always
    # the constant ~10 regardless of text content.
    # NEW: Use raw emotion word counts normalized by total word count so each
    # emotion dimension is independent. A text with more fear words scores
    # higher on fear, and texts with more emotional content score higher overall.
    total_words = max(len(text_object.words), 1)
    raw_scores = text_object.raw_emotion_scores

    # All emotions EmotionalResponse expects — default to 0.0 if absent
    ALL_EMOTIONS = [
        "fear", "anger", "anticipation", "trust", "surprise",
        "positive", "negative", "sadness", "disgust", "joy",
    ]

    # Normalize: (emotion word count / total words) * 100 → [0, 100] per dimension
    emotion_pcts: dict[str, float] = {e: 0.0 for e in ALL_EMOTIONS}
    for emotion, count in raw_scores.items():
        if emotion in emotion_pcts:
            emotion_pcts[emotion] = (count / total_words) * 100

    # Handle the vestigial "anticip" / "anticipation" key inconsistency in NRCLex
    if emotion_pcts["anticipation"] == 0.0 and "anticip" in raw_scores:
        emotion_pcts["anticipation"] = (raw_scores["anticip"] / total_words) * 100

    # FIX #4: Now passes **weights to constructor so config-specified ER
    # dimension weights are used (previously silently dropped, defaulted to 0.1).
    return EmotionalResponse(
        **emotion_pcts,
        **weights,
    )


async def _compute_correctness(
    message: str, original_prompt: str, prompts: Prompts, weights: dict[str, float],
    model: str = "llama3.2",
) -> CorrectnessResponse:
    content = prompts.get_prompt(
        PromptNames.Correctness,
        {"original_prompt": original_prompt, "message": message},
    )
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return CorrectnessResponse(
            # FIX #7: All values through _safe_float. Previously logical_consistency
            # had no conversion at all, and int() would fail on float strings.
            logical_consistency=_safe_float(parsed_response["logical_consistency"]),
            factual_accuracy=_safe_float(parsed_response["factual_accuracy"]),
            contextual_appropriateness=_safe_float(
                parsed_response["contextual_appropriateness"]
            ),
            **weights,
        )
    # FIX #5: Was bare `except:` — now catches Exception only, logs for debugging.
    except Exception as e:
        logger.warning(f"Failed to parse correctness response: {e}")
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
    model: str = "llama3.2",
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
        model=model, messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return ExperientialMatchingResponse(
            knowledge_base_matching=_safe_float(parsed_response["knowledge_base_matching"]),
            historical_responses_matching=_safe_float(
                parsed_response["historical_responses_matching"]
            ),
            **weights,
        )
    except Exception as e:
        logger.warning(f"Failed to parse experiential matching response: {e}")
        return ExperientialMatchingResponse(
            knowledge_base_matching=0.0, historical_responses_matching=0.0, **weights
        )


async def _compute_conflict_information(
    message: str,
    sources: str,
    temporal_info: str,
    prompts: Prompts,
    weights: dict[str, float],
    model: str = "llama3.2",
) -> ConflictInformation:
    content = prompts.get_prompt(
        PromptNames.Conflict_Information,
        {"sources": sources, "message": message, "temporal_info": temporal_info},
    )
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return ConflictInformation(
            internal_consistency=_safe_float(parsed_response["internal_consistency"]),
            source_agreement=_safe_float(parsed_response["source_agreement"]),
            temporal_stability=_safe_float(parsed_response["temporal_stability"]),
            **weights,
        )
    except Exception as e:
        logger.warning(f"Failed to parse conflict information response: {e}")
        return ConflictInformation(
            internal_consistency=0.0,
            source_agreement=0.0,
            temporal_stability=0.0,
            **weights,
        )


async def _compute_problem_importance(
    original_prompt: str, prompts: Prompts, weights: dict[str, float],
    model: str = "llama3.2",
) -> ProblemImportance:
    content = prompts.get_prompt(
        PromptNames.Problem_Importance, {"original_prompt": original_prompt}
    )
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": content}]
    )
    try:
        parsed_response = json.loads(response.message.content)
        return ProblemImportance(
            potential_consequences=_safe_float(parsed_response["potential_consequences"]),
            temporal_urgency=_safe_float(parsed_response["temporal_urgency"]),
            scope_of_impact=_safe_float(parsed_response["scope_of_impact"]),
            **weights,
        )
    except Exception as e:
        logger.warning(f"Failed to parse problem importance response: {e}")
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
