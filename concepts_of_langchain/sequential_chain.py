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

prompt1=PromptTemplate(
    template="Generate a report on {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Summarize the report in 5 lines,\n {Text}",
    input_variables=[]
)

parser=StrOutputParser()

chain=prompt1|model|parser|prompt2|model|parser
result=chain.invoke({'topic':'Python programming language'})

print(result)

# to display the chain
chain.get_graph().print_ascii()