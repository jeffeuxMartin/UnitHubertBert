from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
from transformers import RobertaConfig

import torch
from pathlib import Path

from torch.utils.data import Dataset

from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset


def jeff_init_tokenizer(tokenizer_file=None):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="newfastbpe/jeff-hubert.json")
    # or use the RobertaTokenizer from `transformers` directly.
            
    tokenizer.unk_token = "[UNK]"
    # tokenizer.unk_token_id = 0
    tokenizer.cls_token = "[CLS]"
    # tokenizer.cls_token_id = 1
    tokenizer.bos_token = "[CLS]"
    # tokenizer.bos_token_id = 1
    tokenizer.sep_token = "[SEP]"
    # tokenizer.sep_token_id = 2
    tokenizer.eos_token = "[SEP]"
    # tokenizer.eos_token_id = 2
    tokenizer.pad_token = "[PAD]"
    # tokenizer.pad_token_id = 3
    tokenizer.mask_token = "[MASK]"
    # tokenizer.mask_token_id = 4
    
    return tokenizer
    
class JeffHubDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = jeff_init_tokenizer()

        self.examples = []

        # src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("*-train.txt")
        src_files = [Path('./data/train-clean-100.wordcollunit.txt')]
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

#############


config = RobertaConfig(
    vocab_size=50_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)



model = RobertaForMaskedLM(config=config)

tokenizer = jeff_init_tokenizer()
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='./data/train-clean-100.wordcollunit.txt',
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir="./JeffHubBERTo",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=64,
    save_steps=1_000,
    save_total_limit=5,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
