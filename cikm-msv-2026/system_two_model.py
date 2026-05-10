from enum import StrEnum, auto

import ollama
from pydantic import BaseModel

from metacognitive import MetacognitiveVector, compute_metacognitive_state_vector
from prompts import PromptNames, Prompts
from system_communication_objects import SystemTwoRequest


class NodeRole(StrEnum):
    Domain_Expert = auto()
    Critic = auto()
    Evaluator = auto()
    Generalist = auto()
    Synthesizer = auto()


class NodeResponse(BaseModel):
    node_role: str
    node_response: str
    node_msv: MetacognitiveVector

    class Config:
        frozen = True


class SystemTwoResponse(BaseModel):
    system_two_response: str | None
    metacognitive_vector: MetacognitiveVector | None
    node_responses: list[NodeResponse] | None


class Node:
    role: NodeRole = NodeRole.Generalist
    # TODO adjust weights!
    role_weights: dict[NodeRole, dict[str, float]] = {
        NodeRole.Domain_Expert: {
            "emotional_response": 0.0,
            "correctness": 0.7,
            "experiential_matching": 0.0,
            "conflict_information": 0.1,
            "problem_importance": 0.2,
        },
        NodeRole.Critic: {
            "emotional_response": 0.0,
            "correctness": 0.5,
            "experiential_matching": 0.05,
            "conflict_information": 0.4,
            "problem_importance": 0.05,
        },
        NodeRole.Evaluator: {
            "emotional_response": 0.0,
            "correctness": 0.4,
            "experiential_matching": 0.0,
            "conflict_information": 0.3,
            "problem_importance": 0.3,
        },
        NodeRole.Generalist: {
            "emotional_response": 0.2,
            "correctness": 0.2,
            "experiential_matching": 0.2,
            "conflict_information": 0.2,
            "problem_importance": 0.2,
        },
        NodeRole.Synthesizer: {
            "emotional_response": 0.0,
            "correctness": 0.25,
            "experiential_matching": 0.25,
            "conflict_information": 0.25,
            "problem_importance": 0.25,
        },
    }

    def get_role_preferences(
        self, system_one_vector: MetacognitiveVector
    ) -> dict[NodeRole, float]:
        role_preferences: dict[NodeRole, float] = {}
        for role, weights in self.role_weights.items():
            running_value = 0.0
            for vector_name, weight in weights.items():
                # maybe find a better way to do this than a really flexi-typed accessor into ResponseVectors
                running_value += (
                    weight * getattr(system_one_vector, vector_name).calculated_value
                )
            role_preferences[role] = running_value
        return role_preferences

    def assign_role(self, new_role: NodeRole) -> None:
        # TODO? keep role history?
        # TODO? update role weights?
        self.role = new_role

    def get_response(
        self,
        user_prompt: str,
        previous_node_response: str,
        previous_node_role: NodeRole,
        prompts: Prompts,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": prompts.get_prompt(
                    PromptNames(f"{self.role}_system"),
                    context={"previous_node_role": previous_node_role},
                ),
                "thinking": "true",
            },
            {"role": "assistant", "content": previous_node_response},
            {
                "role": "user",
                "content": prompts.get_prompt(
                    PromptNames(f"{self.role}_user"),
                    context={"user_prompt": user_prompt},
                ),
            },
        ]

        response = ollama.chat(model="llama3.2", messages=messages)
        return response.message.content


class SystemTwo:
    def __init__(self):
        self.nodes = [Node(), Node()]
        self.taken_roles: dict[NodeRole, Node | None] = {
            NodeRole.Domain_Expert: None,
            NodeRole.Critic: None,
            NodeRole.Evaluator: None,
            NodeRole.Generalist: None,
            NodeRole.Synthesizer: None,
        }

    def _reset_taken_nodes(self) -> None:
        self.taken_roles: dict[NodeRole, Node | None] = {
            NodeRole.Domain_Expert: None,
            NodeRole.Critic: None,
            NodeRole.Evaluator: None,
            NodeRole.Generalist: None,
            NodeRole.Synthesizer: None,
        }

    def _transition_nodes(self, system_one_vector: MetacognitiveVector):
        self._reset_taken_nodes()

        for node in self.nodes:
            # TODO: manage role balance w/ Hungarian algo, right now use first available
            role_preferences = node.get_role_preferences(system_one_vector)
            sorted_role_preferences = [
                role
                for role, _ in sorted(
                    role_preferences.items(), key=lambda r: r[1], reverse=True
                )
            ]
            for role in sorted_role_preferences:
                if self.taken_roles[role] == None:
                    self.taken_roles[role] = node
                    node.assign_role(role)
                    break

    async def get_response(
        self,
        user_prompt: str,
        system_one_response: str,
        system_one_vector: MetacognitiveVector,
        prompts: Prompts,
        weights,
    ) -> SystemTwoResponse:

        messages = [
            {
                "role": "system",
                "content": prompts.get_prompt(
                    PromptNames.System_Two_System, context={}
                ),
                "thinking": "true",
            },
            {"role": "assistant", "content": system_one_response},
        ]

        self._transition_nodes(system_one_vector)

        role_responses: list[NodeResponse] = []
        previous_response = system_one_response
        previous_role = "system one"
        synthesizer_response: str | None = None
        synthesizer_msv: MetacognitiveVector | None = None
        for role, node in self.taken_roles.items():
            if node:
                node_response = node.get_response(
                    user_prompt, previous_response, previous_role, prompts
                )

                state = await compute_metacognitive_state_vector(
                    prompts, weights, node_response, previous_response
                )
                role_responses.append(
                    NodeResponse(
                        node_role=role, node_response=node_response, node_msv=state
                    )
                )
                if role == NodeRole.Synthesizer:
                    synthesizer_response = node_response
                    synthesizer_msv = state

                previous_response = node_response
                previous_role = role
                messages.append({"role": "assistant", "content": node_response})
        if not synthesizer_response:
            messages.append(
                {
                    "role": "user",
                    "content": prompts.get_prompt(
                        PromptNames.System_Two_User,
                        context={"user_prompt": user_prompt},
                    ),
                }
            )

            overall_system_two_response = ollama.chat(
                model="llama3.2", messages=messages
            ).message.content
            state = await compute_metacognitive_state_vector(
                prompts, weights, overall_system_two_response, system_one_response
            )
        else:
            overall_system_two_response = synthesizer_response
            state = synthesizer_msv

        return SystemTwoResponse(
            node_responses=role_responses,
            system_two_response=overall_system_two_response,
            metacognitive_vector=state,
        )


system_two = SystemTwo()


async def get_response(system_two_request: SystemTwoRequest) -> SystemTwoResponse:
    return await system_two.get_response(
        system_two_request.user_prompt,
        system_two_request.system_one_response,
        system_two_request.metacognitive_vector,
        system_two_request.prompts,
        system_two_request.weights,
    )
