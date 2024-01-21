from transformer_question_answering.logging.exception import CustomException
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from transformer_question_answering.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from transformer_question_answering.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from transformer_question_answering.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from transformer_question_answering.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
import sys

STAGE_NAME = "Data Ingestion stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)


STAGE_NAME = "Data Validation stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)


STAGE_NAME = 'Data Transformation stage'
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)

STAGE_NAME = 'Model Training stage'
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)

STAGE_NAME = 'Model Evaluation stage'
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise CustomException(e, sys)