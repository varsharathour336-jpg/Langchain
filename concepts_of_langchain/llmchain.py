from langcahin.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.chains import LLMChain

# model initialize
model=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# create a propmt template
prompt=PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

# create the chain
chain=LLMChain(llm=model, prompt=prompt)
output=chain.run(topic="the benefits of meditation")

print(output)