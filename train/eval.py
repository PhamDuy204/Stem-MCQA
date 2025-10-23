import torch
import argparse

from transformers import GenerationConfig
from utils import *

from tqdm.auto import tqdm
from huggingface_hub import login

login('your_token')
from datasets import load_dataset 

def compute_accuracy(model,ds,tokenizer,generation_config):
    labels= list(ds['chat_template'])
    questions=list(ds['chat_benchmark'])
    correct_ans=0
    for i in tqdm(range(len(labels))):
        label = labels[i]
        tokenized=tokenizer(questions[i],return_tensors='pt').to(model.device)
        with torch.no_grad():
            token_ans= model.generate(**tokenized, generation_config=generation_config,use_cache=True)
            ans = tokenizer.batch_decode(token_ans,skip_special_tokens=True)[0]
        _,_,ans_i=ans.partition('### Answer:')
        _,_,label_i=label.partition('### Answer:')
        ans_i=ans_i.strip('\n').strip().split('\n')[0].lower()
        label_i=label_i.strip('\n').strip().split('.')[0].strip()[-1].lower()
        if '.' in ans_i:
            ans_i=ans_i.split('.')[0].strip('\n').strip()[-1]
        else:
            ans_i=ans_i.strip('\n').strip()[0]
        if (ans_i not in ['a','b','c','d','e']):
            print(ans)
            print(ans_i)
        if ans_i==label_i:
            correct_ans+=1 
    return correct_ans/len(labels)

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    ds = load_dataset("mvujas/stem_mcqa_questions")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side='right'
    tokenizer.pad_token = tokenizer.eos_token

    preprocessed_ds=ds['train']\
        .map(lambda x: generate_prompt_and_tokenize(x,tokenizer),batched=True,num_proc=8)
    generation_config = GenerationConfig(
        temperature=0.01,
        repition_penalty=1.3,
        eos_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=2,
        do_sample=False,
        max_new_tokens=50,
    )
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--lora_path", type=str, default=None, help="Lora weigh path")
    args = parser.parse_args()
    
    if args.lora_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model,args.lora_path)
    score=compute_accuracy(model,preprocessed_ds,tokenizer,generation_config)
    print(score)
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write(f"{score}")
    return score

if __name__ =='__main__':
    main()



