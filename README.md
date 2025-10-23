# 📌[PYTORCH] Fine-tuning LLaMA 3.2 1B Instruct for STEM MCQA
## 📖 Introduction

The goal of this project is to fine-tune the LLaMA 3.2 1B Instruct model for STEM multiple-choice question answering (MCQA). While large language models can generate general knowledge responses, they often struggle with domain-specific reasoning in STEM subjects. By fine-tuning on a curated dataset of STEM questions, this project aims to improve the model's ability to understand questions, evaluate multiple-choice options, and select the correct answer.

This approach can be useful for educational tools, automated tutoring systems, and research in AI-driven STEM question answering.


## 📊 Dataset
The collected dataset consists of 2.36k questions in mathematics, physics, chemistry, biology, computer science, and technical sciences.

📂 **Download**: [HugginggFace](https://huggingface.co/datasets/mvujas/stem_mcqa_questions)  


## ⚙️ How to train my project (GPU >=12GB is required)
### Requirements
+ Python>=3.9
+ Cuda >=11.8
### 1 Clone and install libraries
``` bash
git https://github.com/PhamDuy204/Stem-MCQA

pip install -r requirements.txt

hf auth login --token 'your hf access token'
```
### 2 Train Model
```bash
python3 ./train/train.py
```

### 3 Evaluate Model (paste your hf_token into ./train/eval.py)
```bash
python3 ./train/eval.py --lora_path 'None if you want to use base model'
```

## 🧾 Demo

### Step 1: Run the project with Docker Compose (paste your hf_token into docker-compose.yaml file)
```bash
docker compose up --build 
```
### Step 2: Run Fast API
```bash
uvicorn backend.main:app --port 8001
```
### Step 3: Streamlit Demo
```bash
streamlit run demo.py
```
### UI
![Alt text](/assets/a.png)
