from pathlib import Path
from transformer_question_answering.constants import *
from transformer_question_answering.utils.common import read_yaml, create_directories
from transformer_question_answering.entity import DataIngestionConfig
from transformer_question_answering.entity import DataValidationConfig
from transformer_question_answering.entity import DataTransformationConfig
from transformer_question_answering.entity import ModelTrainerConfig

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
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
                root_dir = config.root_dir,
                data_path = config.data_path,
                tokenizer_path = config.tokenizer_path
        )
        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
                root_dir = config.root_dir,
                data_path = config.data_path,
                model_ckpt_path = config.model_ckpt_path,
                num_train_epochs = params.num_train_epochs,
                warmup_steps = params.warmup_steps,
                per_device_train_batch_size = params.per_device_train_batch_size,
                per_device_eval_batch_size = params.per_device_eval_batch_size,
                weight_decay = params.weight_decay,
                logging_steps = params.logging_steps,
                evaluation_strategy = params.evaluation_strategy,
                eval_steps = params.eval_steps,
                save_steps = params.save_steps,
                gradient_accumulation_steps = params.gradient_accumulation_steps
        )
        return model_trainer_config

if __name__ == '__main__':
    try:
        obj = ConfigurationManager()
        obj.get_data_ingestion_config()
    except Exception as e:
        raise e