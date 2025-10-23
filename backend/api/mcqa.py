from fastapi import APIRouter
from backend.models.mcqa import QuestionRequest
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.template import MCQA_TEMPLATE
from  backend.api.llm import mcqa_chat_llm


router = APIRouter()


class QAOutputParser(StrOutputParser):
    """Parser that extracts just the answer choice from the medical QA response."""
    
    def parse(self, text: str) -> str:
        return text

@router.post("/mcqa")
def answer_medical_question(request: QuestionRequest):
    # if len(request.choices) != 4:
    #     raise HTTPException(status_code=400, detail="Invalid number of choices provided.")
    # print('dasdad')
    prompt = PromptTemplate(
        template=MCQA_TEMPLATE,
        input_variables=["question", "choices"]
    )

    # print(prompt.format(question=request.question,choices=request.choices),flush=True)
    chain = prompt | mcqa_chat_llm | QAOutputParser()
    # print('dasda')
    result = chain.invoke({
        "question": request.question,
        "choices":request.choices
    })
    
    return {
        'response':result
    }