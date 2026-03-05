from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()
model=ChatOpenAI()
parser=StrOutputParser()
parser2=PydanticOutputParser(object=Feedback)
class Feedback(BaseModel):
    sentiment:Literal["Positive","Negative"]=Field(description="The sentiment of the feedback, either Positive or Negative.")

Prompt1=PromptTemplate(
    template="Classify the sentiment of the following text into positive or negative: {Feedback} \n {format_instruction}",
    input_variables=['Feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
prompt2=PromptTemplate(
    template="Write an appropriate response for the positive feedback: {Feedback}",
    input_variables=['Feedback']
)
prompt3=PromptTemplate(
    template="Write an appropriate response for the negative feedback: {Feedback}",
    input_variables=['Feedback']
)
classification_chain=Prompt1|model|parser
base_chain=RunnableBranch(
    (lambda x: x["sentiment"] == "Positive", prompt2|model|parser),
    (lambda x: x["sentiment"] == "Negative", prompt3|model|parser),
    RuunableLambda(lambda x: "Invalid sentiment")
)
chain=classification_chain|base_chain

result=chain.invoke({"Feedback":"I love the new design of your website!"})
print(result)

chain.get_graph().print_ascii()