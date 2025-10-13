import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files(dataset="matthewjansen/ucf101-action-recognition", path="DATA/UCF101", quiet=False, unzip=True, force=True)