from huggingface_hub import HfApi, upload_file
api = HfApi()

api.upload_file(
    path_or_fileobj="delivery_model.pkl",
    path_in_repo="delivery_model.pkl",
    repo_id="leo-bsb/Delivery-Time-Predictor",
    repo_type="space"
)
