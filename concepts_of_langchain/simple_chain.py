from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="Give me 5 intersting facts about the {topic}",
    input_variables=['topic']
)
parser=StrOutputParser()

chain=template|model|parser

final_result=chain.invoke({'topic':'Python programming language'})
print(final_result)