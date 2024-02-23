import os
import chromadb
import autogen
import random
import json
from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.groupchat import GroupChat
import autogen
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import getpass
from langchain.document_loaders import PyPDFLoader


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-3.5-turbo", "gpt-35-turbo", "gpt-35-turbo-0613", "gpt-4", "gpt4", "gpt-4-32k"],
    },
)


llm_config_proxy = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "request_timeout": 600
}

# This is a long document we can split up.
with open("commands.txt") as f:
    state_of_the_union = f.read()

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])

os.environ['OPENAI_API_KEY'] = "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"

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('commands.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

from langchain.text_splitter import CharacterTextSplitter

# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )

query = "What is the name of the girl"
# docs = db.similarity_search(query)

docs = PyPDFLoader('uniswap_v3.pdf').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(docs)

vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OpenAIEmbeddings()
)
vectorstore.add_documents(docs)

qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    vectorstore.as_retriever(),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

# Test the QA chain
result = qa(({"question": "What is uniswap?"}))
print(result['answer'])


def answer_uniswap_question(question):
    response = qa({"question": question})
    return response["answer"]


llm_config={
    # "request_timeout": 6000,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
    "functions": [
        {
            "name": "answer_uniswap_question",
            "description": "Answer any Uniswap related questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask in relation to Uniswap protocol",
                    }
                },
                "required": ["question"],
            },
        }
    ],
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "."},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
    function_map={"answer_uniswap_question": answer_uniswap_question}
)

class Config(object):
    MESSAGE = os.environ.get("MESSAGE")


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    user_proxy.initiate_chat(
        assistant,
        message="""
            I'm writing a blog to introduce the version 3 of Uniswap protocol. Find the answers to the 3 questions below and write an introduction based on them.
            
            1. What is Uniswap?
            2. What are the main changes in Uniswap version 3?
            3. How to use Uniswap?
            
            Start the work now.
        """
    )
    return str(user_proxy.chat_messages)


@app.route("/settings")
def get_settings():
    return {"message": app.config["MESSAGE"]}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
