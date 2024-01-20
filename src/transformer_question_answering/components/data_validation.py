import os
import sys
from transformer_question_answering.utils.common import get_size
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.logging.exception import CustomException
from transformer_question_answering.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def validate_all_file_exists(self) -> bool:
        """
        Returns:
            bool: True if all the required files exist in the data directory.
        """
        try:
            validation_status = True
            required_files = self.config.ALL_REQUIRED_FILES
            data_directory = os.path.join('artifacts', 'data_ingestion', 'data')
            
            for file in required_files:
                file_path = os.path.join(data_directory, file)

                if not os.path.exists(file_path):
                    validation_status = False
                    break

            with open(self.config.STATUS_FILE, 'w') as file_object:
                file_object.write(f'Validation status: {validation_status}')

            return validation_status
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)