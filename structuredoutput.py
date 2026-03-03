from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os


load_dotenv()

print("Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
scehema=[
    ResponseSchema(name="fact1",description="the fact1 about the topic"),
    ResponseSchema(name="fact2",description="the fact2 about the topic"),
    ResponseSchema(name="fact3",description="the fact3 about the topic")
]
parser=StructuredOutputParser.from_response_schemas(scehema)
template=PromptTemplate(
    template='Give me 3 facts about the topic \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
promt=template.invoke({'topic':'Python programming language'})
result=model.invoke(promt)
parswed_result=parser.parse(result.content)
print(parswed_result)