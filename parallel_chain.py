from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable
load_dotenv()
model1=ChatOpenAI()
model2=ChatAnthropic(model_name="claude-3")

propmt1=PromptTemplate(
    template ="Give me the notes of following report{report}",
    input_variables=['report']
)
propmt2=PromptTemplate(
    template="Generate the quiz from these notes.\n{text}",
    input_variables=['text']

)
prompt3=PromptTemplate(
    template="Merge the notes and quize into a single document.\n {NOTES},{QUIZ}",
    input_variables=['NOTES', 'QUIZ']
)
parser=StrOutputParser()

parallel_chain=Runnable.parallel({
    'notes': propmt1 | model1 | parser,
    'quiz': propmt2 | model2 | parser
})
