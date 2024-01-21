from transformer_question_answering.logging.logger import logging
from transformer_question_answering.logging.exception import CustomException
from transformer_question_answering.config.configuration import ConfigurationManager
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import pipeline
import sys
import torch

class PredictionPipeline:
    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ConfigurationManager().get_model_evaluation_config()
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.config.tokenizer_path)
        trained_model = DistilBertForQuestionAnswering.from_pretrained(self.config.model_path).to(device)
        self.qa_pipe = pipeline("question-answering", model=trained_model, tokenizer=tokenizer, device=device)


    def predict(self, text: str, question: str) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        result = self.qa_pipe(question=question, context=text)

        print('Text')
        print(text)

        print('\nQuestion')
        print(question)

        print('\nAnswer')
        print(result['answer'].capitalize())

        return result['answer'].capitalize()
    

if __name__ == "__main__":
    obj = PredictionPipeline()
    obj.predict(text='The kitchen is east of the hallway. The garden is south of the bedroom.', question="What is east of the hallway?")

