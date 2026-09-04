import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.MlflowClient()

versions = client.search_model_versions('name="ride-duration-predictor"')

for v in versions:
    print("Version:", v.version, "| Stage:", v.current_stage, "| Run ID:", v.run_id)
