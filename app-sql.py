import os
from typing import Any, Dict

import chromadb
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from flask import Flask
import sqlite3
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities.sql_database import SQLDatabase
from langchain.llms import OpenAI


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-3.5-turbo", "gpt-35-turbo", "gpt-35-turbo-0613", "gpt-4", "gpt4", "gpt-4-32k"],
    },
)


# llm_config = {
#     "timeout": 60,
#     "cache_seed": 42,
#     "config_list": config_list,
#     "temperature": 0,
# }

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


def create_table(cursor, create_table_sql):
    """ Execute a SQL statement to create a table. """
    cursor.execute(create_table_sql)


def insert_data(cursor, insert_sql, data):
    """ Insert data into a table using the provided SQL statement. """
    cursor.executemany(insert_sql, data)


database = 'bookstore.db'


with sqlite3.connect(database) as conn:
    cursor = conn.cursor()

    # Create the Books table
    sql_create_books_table = """
        CREATE TABLE IF NOT EXISTS Books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author_id INTEGER,
            publisher_id INTEGER,
            price REAL
        ); """
    create_table(cursor, sql_create_books_table)

    # Create the Authors table
    # Create the Authors table
    sql_create_authors_table = """
        CREATE TABLE IF NOT EXISTS Authors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            biography TEXT
        );"""
    create_table(cursor, sql_create_authors_table)


    # Create the Publishers table
    sql_create_publishers_table = """
        CREATE TABLE IF NOT EXISTS Publishers (
            d INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            address TEXT
        );"""
    create_table(cursor, sql_create_publishers_table)

    # Insert sample data into Books table
    insert_books_sql = "INSERT INTO Books (title, author_id, publisher_id, price) VALUES (?, ?, ?, ?)"
    books_data = [
        ('To Kill a Mockingbird', 1, 1, 10.99),
        ('1984', 2, 2, 8.99)
    ]
    insert_data(cursor, insert_books_sql, books_data)

    # Insert sample data into Authors table
    insert_authors_sql = "INSERT INTO Authors (name, biography) VALUES (?, ?)"
    authors_data = [
        ('Harper Lee', 'Author of To Kill a Mockingbird'),
        ('George Orwell', 'Author of 1984')
    ]
    insert_data(cursor, insert_authors_sql, authors_data)

    # Insert sample data into Publishers table
    insert_publishers_sql = "INSERT INTO Publishers (name, address) VALUES (?, ?)"
    publishers_data = [
        ('Publisher A', '123 Street, City'),
        ('Publisher B', '456 Avenue, City')
    ]
    insert_data(cursor, insert_publishers_sql, publishers_data)

    conn.commit()
    cursor.close()

# Use an absolute path for the database
os.environ["OPENAI_API_KEY"] = "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"
database_path = 'C:/Users/patri/IdeaProjects/agent-swarm-autogen/bookstore.db'
db = SQLDatabase.from_uri(f'sqlite:///{database_path}')
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


def generate_llm_config(tool: Any) -> Dict[str, Any]:
    """
    Generate a Function schema from a LangChain's Agent Tool.

    Args:
    tool (Any): The tool to generate the schema for.

    Returns:
    Dict[str, Any]: The generated function schema.
    """
    function_schema = {
        "name": tool.name.lower().replace(' ', '_'),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    # Assuming tool.args is a dictionary or None
    if tool.args:
        function_schema["parameters"]["properties"] = tool.args

    return function_schema


# Usage with Langchain Tool Bridge
tools = [generate_llm_config(tool) for tool in toolkit.get_tools()]  # List comprehension
function_map = {tool.name: tool._run for tool in toolkit.get_tools()}  # Dictionary comprehension


# config_list = [{
#     'model': 'gpt-4-1106-preview',
#     'api_key': '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'
# }]


# Construct the llm_config
llm_config = {
    "functions": tools,
    "config_list": config_list,  # Assuming you have this defined elsewhere
    "timeout": 120,
}


user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "tmp", "use_docker": False},
)

# Register the tool and start the conversation
user_proxy.register_function(
    function_map = function_map
)

Sql_chatbot = autogen.AssistantAgent(
    name="Sql_chatbot",
    system_message="""Please adhere strictly to the predefined functions for all coding assignments. 
                    Once you have completed the task, kindly respond with the word 'TERMINATE' to signify its completion..""",
    llm_config=llm_config,
)


class Config(object):
    MESSAGE = os.environ.get("MESSAGE")


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    user_proxy.initiate_chat(
        Sql_chatbot,
        message="how many duplicate books in table",
        llm_config=llm_config,
    )
    return 'Hello World!!!!' + Sql_chatbot.last_message().get('content')


@app.route("/settings")
def get_settings():
    return {"message": app.config["MESSAGE"]}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
