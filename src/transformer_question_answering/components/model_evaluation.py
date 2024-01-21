from transformer_question_answering.entity import ModelEvaluationConfig
from datasets import load_from_disk, load_metric
from transformer_question_answering.utils.common import compute_metrics
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
import pandas as pd
import torch
from tqdm import tqdm


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """
            Split the dataset into smaller batches that can be processed simultaneously.
            Yield successive batch-sized chunks from list of elements
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    
    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer,
                                    batch_size=16, 
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),column_text='article',
                                    column_summary='highlights'):
        
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):

            inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')
            
            summaries = model.generate(input_ids = inputs['input_ids'].to(device),
                                     attention_mask = inputs['attention_mask'].to(device),
                                     length_penalty=0.8,
                                     num_beams=8,
                                     max_length=128)
            ''' parameter for length_penality ensures that the model does not generate sequences that are too long.'''

            # Finally, we decode the generated texts by replacing the token, and add the decoded
            # texts with references to the metric.

            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)            
                                 for s in summaries]
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        # Finally compute and return the ROUGE score
        score = metric.compute()

        return score
    
    def evaluate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenzier = DistilBertTokenizerFast.from_pretrained(self.config.tokenizer_path)
        trained_model = DistilBertForQuestionAnswering.from_pretrained(self.config.model_path).to(device)

        # Loading the data
        # qa_dataset = load_from_disk(self.config.data_path)
        
        question, text = 'What is east of the hallway?','The kitchen is east of the hallway. The garden is south of the bedroom.'

        input_dict = tokenzier(text, question, return_tensors='pt')

        input_ids = input_dict['input_ids'].to(device)
        attention_mask = input_dict['attention_mask'].to(device)

        outputs = trained_model(input_ids, attention_mask=attention_mask)

        start_logits = outputs[0]
        end_logits = outputs[1]

        all_tokens = tokenzier.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        answer = ' '.join(all_tokens[torch.argmax(start_logits, 1)[0] : torch.argmax(end_logits, 1)[0]+1])

        print(question, answer.capitalize())


        # score = self.calculate_metric_on_test_ds(
        #     qa_dataset['test'], 
        #     metric = compute_metrics, 
        #     model=trained_model, 
        #     tokenizer=tokenzier, 
        #     batch_size=2, 
        # ) 

        # print(score)

