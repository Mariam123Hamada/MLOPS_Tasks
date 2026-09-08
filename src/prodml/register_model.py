# #this si the section of the register model

# import mlflow
# from dotenv import load_dotenv

# load_dotenv()


# ############################### Resgister Model ###################
# mlflow.set_tracking_uri("http://localhost:5000")

# RUN_ID="980886505b764ee0af18ba4f98b0496c"

# MODEL_NAME = "ride-duration-predictor"

# model_uri = f"runs:/{RUN_ID}/model"

# result = mlflow.register_model(
#     model_uri=model_uri,
#     name=MODEL_NAME,
# )

# print(f"Registered model: {result.name}")
# print(f"Version: {result.version}")

# ##################################################################################################

# ######################## Staging Model #########################################################

# # this is the section of prompted model to staging because in the ui the choice stage is deprectaed.
# # Stage
# from mlflow import MlflowClient

# client = MlflowClient()

# model_name = "ride-duration-predictor"
# model_version = "1"

# # Move Version 1 to Staging
# client.transition_model_version_stage(
#     name=model_name,
#     version=model_version,
#     stage="Staging",
#     archive_existing_versions=False,
# )

# print(f"Model {model_name} version {model_version} is now in Staging.")

# ############################################################################################################


# ################################### Prompte Model To Production ###############################################


# #lets prompote it to the Productaion stage

# from mlflow import MlflowClient

# client = MlflowClient()

# client.transition_model_version_stage(
#     name="ride-duration-predictor",
#     version="11",
#     stage="Production",
#     archive_existing_versions=True,
# )

# print("Version 1 promoted to Production.")

# ################################################################################################


# ##################3 Register the Worest Model #############################################
# # The Worest Model:
# Run_Worest_Model_ID="d35a5bb181b34f90ae50b84eb72a4739"
# model_uri = f"runs:/{Run_Worest_Model_ID}/model"

# result = mlflow.register_model(
#     model_uri=model_uri,
#     name="ride-duration-predictor",
# )

# print(result.version)


import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "ride-duration-predictor"
RUN_ID = (
    "75c9e2973c9145bcb6742000ec64de27"  # Update this with the latest training run ID
)

client = MlflowClient()

# Register model from the latest training run
model_uri = f"runs:/{RUN_ID}/model"
result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"Registered model: {result.name}, Version: {result.version}")

# Promote the just-registered version to Production
latest_version = result.version
print(f"Promoting version {latest_version} to Production...")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True,
)

print(f"✅ Version {latest_version} promoted to Production.")
