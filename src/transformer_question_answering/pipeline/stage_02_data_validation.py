import os
import sys
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.logging.exception import CustomException
from transformer_question_answering.config.configuration import ConfigurationManager
from transformer_question_answering.components.data_validation import DataValidation



class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_file_exists()
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataValidationTrainingPipeline()
    obj.main()