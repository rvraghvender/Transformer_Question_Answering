from transformer_question_answering.entity import ModelEvaluationConfig
from datasets import load_from_disk
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformer_question_answering.utils.common import compute_metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import pandas as pd
import torch
from tqdm import tqdm


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config

    
    def calculate_metric_on_test_ds(self, test_dataset, model, tokenizer,
                                    batch_size=16, 
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        model.to(device)
        model.eval()


        all_start_positions = []
        all_end_positions = []
        all_start_preds = []
        all_end_preds = []

        for batch in DataLoader(test_dataset, batch_size=batch_size):
 
            inputs = tokenizer(batch['sentences'], batch['question'],  return_tensors="pt", truncation=True, padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Capturing start and end position details from test_dataset example
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Assuming that 'outputs' contains start_logits and end_logits
            start_preds = torch.argmax(outputs.start_logits, dim=-1)
            end_preds = torch.argmax(outputs.end_logits, dim=-1)

            all_start_positions.extend(start_positions.tolist())
            all_end_positions.extend(end_positions.tolist())
            all_start_preds.extend(start_preds.tolist())
            all_end_preds.extend(end_preds.tolist())

        # Compute f1 score for start and end position between original and predicted positions
        f1_start = f1_score(all_start_positions, all_start_preds, average='macro')
        f1_end = f1_score(all_end_positions, all_end_preds, average='macro')

        # Compute accuracy score for start and end position between original and predicted positions
        accuracy_start = accuracy_score(all_start_positions, all_start_preds)
        accuracy_end = accuracy_score(all_end_positions, all_end_preds)

        return {'f1_start': f1_start, 'f1_end': f1_end, 
                'accuracy_start': accuracy_start, 'accuracy_end': accuracy_end}

    
    def evaluate(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = DistilBertTokenizerFast.from_pretrained(self.config.tokenizer_path)
        trained_model = DistilBertForQuestionAnswering.from_pretrained(self.config.model_path).to(device)

        # Loading the data
        qa_dataset = load_from_disk(self.config.data_path)
        test_ds = qa_dataset['test']
        test_ds.set_format(type='pt')

        scores = self.calculate_metric_on_test_ds(test_ds,  trained_model, tokenizer)

        df = pd.DataFrame(scores, index=['babl_dataset'])
        df.to_csv(self.config.metric_file_name, index=False)