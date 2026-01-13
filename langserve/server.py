from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0)

# Create FastAPI app
app = FastAPI(title="Langchain server",version="0.1", description="Langchain server with Groq LLM")

genric_template = "Translate the following into {language}:" 

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(genric_template),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)


parser = StrOutputParser()

# Create Chain
chain = prompt | model | parser

# Add LangServe routes
add_routes(app, chain, path="/translate")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)