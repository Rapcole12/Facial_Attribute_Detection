import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

#just to download the zip file - not necessary I will attach first 20,000 images to git
dataset = "jessicali9530/celeba-dataset"
download_dir = "celeba_data"
api.dataset_download_files(dataset, path=download_dir, unzip=False)
