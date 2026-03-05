from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from dotenv import load_dotenv
import os


load_dotenv()

print("Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()
template=PromptTemplate(
    template='Give me the name and age of the fictional person \n {format_instruction}',
    # Give me the name and age of the fictional person
    #  by {format_instructions}Return a JSON object.
    input_variables=[],
#   partial variable means it desnot take value at runtime
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt=template.format()
# result=model.invoke(prompt)
# print(result)

# final_result=parser.parse(result.content)
chain=template|model|parser
final_result=chain.invoke({})
print(final_result)