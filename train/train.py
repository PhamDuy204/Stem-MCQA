from utils import *

from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig,\
                        TrainingArguments, Trainer, DataCollatorForLanguageModeling

from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training 

import torch
from transformers import TrainerCallback
from rich.console import Console
from rich.table import Table

class LogLossCallback(TrainerCallback):
    def __init__(self):
        self.console = Console()
        self.table = Table(show_header=True, header_style="bold magenta")
        self.table.add_column("Step", justify="right")
        self.table.add_column("Training Loss", justify="right")
        self.table.add_column("Eval Loss", justify="right")
        self.logged_steps = set()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            if step not in self.logged_steps:
                loss = logs["loss"]
                self.table.add_row(str(step), f"{loss:.6f}", f"-")
                self.logged_steps.add(step)

            if step % 10 == 0:
                self.console.print(self.table)

def train():    
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side='right'
    tokenizer.pad_token = tokenizer.eos_token

    ds = get_ds()

    preprocessed_ds=ds['train'].map(lambda x: generate_prompt_and_tokenize(x,tokenizer),remove_columns=['question', 'answer', 'explanation', 'field'],batched=True,num_proc=8)
    
    train_set,test_set=preprocessed_ds.train_test_split(test_size=0.1).values()
    train_set.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    test_set.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True,
        quantization_config=quantization_config, 
        dtype="auto"
    )

    model=prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_disable()

    lora_config=LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            'q_proj','o_proj','v_proj','k_proj','gate_proj','up_proj','down_proj'
        ],
        lora_dropout=0.05,
        bias='none'
    )

    model=get_peft_model(model,lora_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        batch_eval_metrics=False,
        gradient_accumulation_steps=1,
        num_train_epochs=4,
        learning_rate=1e-4,
        fp16=True,
        save_total_limit=3,
        logging_steps=50,
        output_dir="llama3-1b-mcqa-1",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="steps",
        eval_steps=50,
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    trainer=Trainer(
        model=model,
        # tokenizer=tokenizer,

        train_dataset=train_set,
        eval_dataset=test_set,
        args=training_args,
        data_collator=data_collator,
        callbacks=[LogLossCallback()]
    )
    
    model.config.use_cache=False
    model.enable_input_require_grads()
    trainer.train()

if __name__ == '__main__':
    train()


