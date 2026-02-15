from pydantic import BaseModel, Field

class Experiment(BaseModel):
    id: str
    prompts: list[str]

class Experiments(BaseModel):
    experiments: list[Experiment]


class CompletedExperiment(BaseModel):
    experiment_id: str
    session_id: str
    errors: list[str]
    experiment_start: str
    duration_seconds: int


class CompletedExperiments(BaseModel):
    completed_experiments: list[CompletedExperiment] = Field(default=[])


class SystemOnePrompt(BaseModel):
    user_input: str

class SystemOneResponse(BaseModel):
    response: str
    session_id: str
