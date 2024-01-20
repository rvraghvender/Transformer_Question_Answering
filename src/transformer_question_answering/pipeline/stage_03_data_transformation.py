import sys
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.logging.exception import CustomException
from transformer_question_answering.components.data_transformation import DataTransformation
from transformer_question_answering.config.configuration import ConfigurationManager

class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config =  ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.convert()
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataTransformationTrainingPipeline()
    obj.main()