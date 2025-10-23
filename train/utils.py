from datasets import load_dataset 

def get_ds():
    ds = load_dataset("mvujas/stem_mcqa_questions")
    return ds

def generate_conversation(sample):
    bef,opts= sample['question'].replace('**Options**','options').replace('Options','options').split('options:')
    opts='\n'.join([opt.replace('\n','').strip() for opt in opts.replace('\n\n','\n').split('\n')[1:]])
    ans = sample['answer']
    ans_with_context=[opt for opt in opts.split('\n') if opt[0].lower()==ans.lower()][0]
    exp = sample['explanation']+f' So {ans} is correct answer.'
    messages = [
        {
            'role':'system',
            'content':'''You are a helpful assistant with extensive knowledge of mathematics, computer science, engineering, biology, physics and chemistry.Respond safely and accurately.'''
        },
        {
            'role':'user',
            'content':'''### Question:\n{bef}\n\n### Choices:\n{opts}\n\nRespond ONLY with the letter and full text of correct answer choice.
            '''.format(bef=bef.strip('\n'),opts=opts.strip('\n'))
        },
        {
            'role':'assistant',
            'content':'''### Answer:\n{ans_with_context}\n\n### Explanation:\n{exp}\n\nRespond correct answer choice and explain accurately.
            '''.format(ans_with_context=ans_with_context.strip('\n'),exp=exp.strip('\n'))
        }
    ]
    return messages

def generate_prompt_and_tokenize(samples,tokenizer):
    questions,answers,explanations,_=samples.values()
    sample_lst = [{'question':question,
                  'answer':answer,
                  'explanation':explanation} for (question,answer, explanation) in zip(questions,answers,explanations)]
    conversations = [tokenizer.apply_chat_template(generate_conversation(sample),tokenize=False) for sample in sample_lst
    ]
    conversations_benchmark = [''.join(tokenizer.apply_chat_template(generate_conversation(sample),tokenize=False).partition('### Answer:')[:2]) for sample in sample_lst
    ]
    samples['chat_template']=conversations
    samples['chat_benchmark']=conversations_benchmark
    tokenized_prompt = tokenizer(conversations, truncation=True, padding='max_length', max_length=768)
    samples['input_ids']=tokenized_prompt['input_ids']
    samples['attention_mask']=tokenized_prompt['attention_mask']
    samples['labels']=samples['input_ids'].copy()
    return samples