from transformers import TrainingArguments, Trainer
from transformers import DistilBertForQuestionAnswering
from transformer_question_answering.entity import ModelTrainerConfig
from transformer_question_answering.utils.common import compute_metrics
from transformers import DistilBertTokenizerFast
from transformers import EarlyStoppingCallback
import torch
from datasets import load_from_disk
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DistilBertForQuestionAnswering.from_pretrained(self.config.model_ckpt_path).to(device)
        tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join('artifacts', 'data_transformation','tokenizer'))

        qa_dataset = load_from_disk(os.path.join('artifacts', 'data_transformation', 'data'))
        train_ds = qa_dataset['train']
        test_ds = qa_dataset['test']

        columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']
        train_ds.set_format(type='pt', columns=columns_to_return)
        test_ds.set_format(type='pt', columns=columns_to_return)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.root_dir, 'results'),           # output directory
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,             # total number of training epochs
            per_device_train_batch_size=self.config.per_device_train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,   # batch size for evaluation
            warmup_steps=self.config.weight_decay,                # number of warmup steps for learning rate scheduler
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps = self.config.eval_steps,
            save_steps = self.config.save_steps,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            load_best_model_at_end=True,
            save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
        )

        trainer = Trainer(
            model=model,                 # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_ds,         # training dataset
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,             # evaluation dataset
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # trainer.cuda.empty_cache()
        trainer.train()

        print(trainer.evaluate(test_ds))

        model.save_pretrained(os.path.join(self.config.root_dir, "distilbert-qa-model"))

        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, 'tokenizer'))
