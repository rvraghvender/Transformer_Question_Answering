from transformer_question_answering.config.configuration import ConfigurationManager
from transformer_question_answering.components.model_trainer import ModelTrainer
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.logging.exception import CustomException
import sys

class ModelTrainerTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = ModelTrainerTrainingPipeline()
    obj.main()