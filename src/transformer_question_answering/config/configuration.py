from pathlib import Path
from transformer_question_answering.constants import *
from transformer_question_answering.utils.common import read_yaml, create_directories
from transformer_question_answering.entity import DataIngestionConfig
from transformer_question_answering.entity import DataValidationConfig

class ConfigurationManager:
    def __init__(self,
                 config_file_path: Path = CONFIG_FILE_PATH,
                 params_file_path: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,
                source_url = config.source_url,
                local_data_file = config.local_data_file,
                unzip_dir = config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
                root_dir = config.root_dir,
                STATUS_FILE = config.STATUS_FILE,
                ALL_REQUIRED_FILES = config.ALL_REQUIRED_FILES
        )
        return data_validation_config

if __name__ == '__main__':
    try:
        obj = ConfigurationManager()
        obj.get_data_ingestion_config()
    except Exception as e:
        raise e