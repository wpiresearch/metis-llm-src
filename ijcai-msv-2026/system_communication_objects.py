from pydantic import BaseModel

from metacognitive import MetacognitiveVector
from prompts import Prompts

class SystemTwoRequest(BaseModel):
    user_prompt: str
    system_one_response: str
    metacognitive_vector: MetacognitiveVector
    prompts: Prompts
    weights: dict[str, dict[str, float]]

