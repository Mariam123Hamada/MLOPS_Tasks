import mlflow
from mlflow import MlflowClient

MODEL_NAME = "ride-duration-predictor"


def promote_if_better(
    candidate_run_id: str,
    metric: str = "mae",
) -> bool:
    client = MlflowClient()

    candidate_run = client.get_run(candidate_run_id)

    candidate_metric = candidate_run.data.metrics.get(metric)

    if candidate_metric is None:
        raise ValueError(f"Candidate run does not contain metric: {metric}")

    production_versions = client.get_latest_versions(
        MODEL_NAME,
        stages=["Production"],
    )

    if not production_versions:
        print("No Production model exists. Promoting candidate.")
        should_promote = True
    else:
        production_version = production_versions[0]

        production_run = client.get_run(production_version.run_id)

        production_metric = production_run.data.metrics.get(metric)

        if production_metric is None:
            raise ValueError(f"Production model does not contain metric: {metric}")

        print(f"Candidate {metric}: {candidate_metric}")
        print(f"Production {metric}: {production_metric}")

        # For MAE, lower is better.
        should_promote = candidate_metric < production_metric

    if not should_promote:
        print("Candidate is not better. No promotion.")
        return False

    model_uri = f"runs:/{candidate_run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Promoted version {registered.version} " f"to Production.")

    return True
