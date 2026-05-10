import argparse
from datetime import datetime, timezone
import logging

import requests


from experiment_model import (CompletedExperiment, 
                              CompletedExperiments, 
                              Experiments, Experiment, 
                              SystemOnePrompt, 
                              SystemOneResponse)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_experiments(url: str, experiments: Experiments):
    logging.info("Starting experiments run")
    completed = CompletedExperiments()
    utc_run_start = datetime.now(timezone.utc)
    
    for experiment in experiments.experiments:
        experiment_id = experiment.id
        utc_experiment_start = datetime.now(timezone.utc)
        
        experiment_session_id: str | None = None
        errors = []
        logging.info(f"Starting experiment {experiment_id}")
        number_of_prompts = len(experiment.prompts)
        
        for prompt_index, prompt in enumerate(experiment.prompts):
            logging.info(f"Sending prompt {prompt_index+1} of {number_of_prompts}")
            try:
                response = requests.post(f"{url}/system1", json=SystemOnePrompt(user_input=prompt).model_dump())
                decoded_response = SystemOneResponse.model_validate(response.json())
                experiment_session_id = decoded_response.session_id
            except Exception as e:
                response_code = response.status_code if response else -1
                errors.append(f"{prompt_index} - '{response_code}' - '{e}'")
        
        utc_experiment_end = datetime.now(timezone.utc)
        duration = (utc_experiment_end - utc_experiment_start).seconds
        completed.completed_experiments.append(CompletedExperiment(experiment_id=experiment_id, 
                                                         session_id=experiment_session_id, 
                                                         errors=errors,
                                                         experiment_start=utc_experiment_start.strftime("%Y-%m-%d_%H_%M_%S"),
                                                         duration_seconds=duration))
        logging.info(f"Experiment completed: {experiment_id}")
        requests.post(f"{url}/reset")
    
    formatted_datetime = utc_run_start.strftime("%Y-%m-%d_%H_%M")
    completed_experiments_file_name = f"completed_experiments_{formatted_datetime}.json"
    logging.info(f"Run completed saving results to {completed_experiments_file_name}")
    with open(completed_experiments_file_name, "w") as completed_file:
        completed_file.write(completed.model_dump_json(indent=4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="")
    parser.add_argument("--experiment_file", type=str, help="Path to the JSON file containing experiments.")

    args = parser.parse_args()

    with open(args.experiment_file, "r") as file:
        experiments = Experiments.model_validate_json(file.read())
    
    run_experiments(args.url, experiments)


if __name__ == '__main__':
    main()
