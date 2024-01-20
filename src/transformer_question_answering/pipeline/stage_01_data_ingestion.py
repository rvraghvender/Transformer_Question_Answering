from transformer_question_answering.config.configuration import ConfigurationManager
from transformer_question_answering.components.data_ingestion import DataIngestion


class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass
        
    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file(google_drive=True)
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    obj = DataIngestionTrainingPipeline()
    obj.main()