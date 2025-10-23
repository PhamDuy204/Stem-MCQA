import streamlit as st
import string
import requests
st.set_page_config(page_title="Question Builder", layout="wide")

st.title("🧩 Question & Choices Builder")
CHOICE_PREFIXES = list(string.ascii_uppercase[:24])

col1, col2 = st.columns(2)
with col1:
    st.subheader("📝 Question")
    question = st.text_area(
        "Questions:",
        height=200,
        placeholder="Ex: What is the capital of France?"
    )
with col2:
    st.subheader("✅ Choices")
    
    number = st.number_input(
        "Insert a number", value=1, placeholder="Type a number..."
    )
    if (int(number)<1) or (int(number)>24):
        st.error("❌ Number of question must greater than 1 and less than 24.")
    choices=[

    ]
    for i in range(int(number)):
        prefix = CHOICE_PREFIXES[i]
        choices.append(f'{prefix}. '+st.text_input(
            f"{prefix})",
            key=f"choice_{i}"
        ))     
st.markdown("---")
if st.button("🚀 Submit"):
    url = "http://localhost:8001/v1/mcqa"

    choice_context='\n'.join(choices).strip('\n').strip()
    data = {
        "question": question,
        "choices": choice_context
    }

    response = requests.post(url, json=data)
    output = response.json()['response']

    correct_opt,_,ex = output.partition('### Explanation:')
    correct_opt=correct_opt.replace('### Answer:','').strip().strip('\n')

    ex=ex.strip('\n').split('\n')[0].strip().strip('\n')
    st.markdown("### 🧠 Model Output")
    ans_col, exp_col = st.columns(2)
    with ans_col:
        st.subheader("✅ Answer")
        st.info(correct_opt if correct_opt else "No answer found.")

    with exp_col:
        st.subheader("💡 Explanation")
        st.success(ex if ex else "No explanation found.")

    print("Response:", response.json())

