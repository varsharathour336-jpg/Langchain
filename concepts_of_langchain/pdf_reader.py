from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

# load the documents
loader=TextLoader('document.txt')
documents=loader.load()

# split the documents into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts=text_splitter.split_documents(documents)

# create the embeddings

vector_store=FAISS.from_documents(texts, OpenAIEmbeddings())

# create the retriever
retriever=vector_store.as_retriever()

# manually retrive the relevant documents
query="What is the main topic of the document?"
retrieved_query=retriever.get_relevant_documents(query)

# combined the retrieved documents into a single Prompt
retrieved_text="\n".join([doc.page_content for doc in retrieved_query])

# create the LLM
llm=OpenAI()

# manually pass the prompt to llm
prompt=f"Based on the following retrieved text, answer the question: {query}\n\n{retrieved_text}"
response=llm(prompt)

# print
print(response)