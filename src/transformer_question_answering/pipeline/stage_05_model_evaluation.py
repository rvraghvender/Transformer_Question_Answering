from transformer_question_answering.config.configuration import ConfigurationManager
from transformer_question_answering.components.model_evaluation import ModelEvaluation
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.logging.exception import CustomException
import sys

class ModelEvaluationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelEvaluationTrainingPipeline()
    obj.main()