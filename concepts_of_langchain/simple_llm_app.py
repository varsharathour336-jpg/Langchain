from langchain.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# initialize the model
model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# create a prompt template

prompt=PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=["topic"]
)

# define the topic
topic = "the benefits of meditation"

# format the prompt with the topic
formatted_prompt = prompt.format(topic=topic)

# call the model with the formatted prompt

blog=model(formatted_prompt)
print(blog)
