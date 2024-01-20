import os
import gdown
import urllib.request as request
import zipfile
from pathlib import Path
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.utils.common import get_size
from transformer_question_answering.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_file(self, google_drive):
        if not os.path.exists(self.config.local_data_file):
            if google_drive:
                output = self.config.local_data_file
                url = self.config.source_url
                download_file = gdown.download(url,output, quiet=False, fuzzy=True) 
            else:
                file_name, headers = request.urlretrieve(
                    url = self.config.source_url,
                    filename = self.config.local_data_file
                )
            logging.info(f'{self.config.local_data_file} download!')
        else:
            logging.info(f'File already exists of size: {get_size(Path(self.config.local_data_file))}')

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
