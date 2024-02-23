import os
import chromadb
import autogen
import random
import json
from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.groupchat import GroupChat
from flask import Flask


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-3.5-turbo", "gpt-35-turbo", "gpt-35-turbo-0613", "gpt-4", "gpt4", "gpt-4-32k"],
    },
)


llm_config = {
    "config_list": config_list,
    "seed": 42,
    "functions":[
        {
            "name": "get_account_id",
            "description": "retrieves the account id for a user given their name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":{
                        "type": "string",
                        "description": "The name of the customer that will be used to lookup the account id"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "get_last_bill_amount",
            "description": "Retrieves the last bill amount for a user for a given account id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id":{
                        "type": "string",
                        "description": "The account id fetched from get_account_id that will be used to lookup the last bill for the customer"
                    }
                },
                "required": ["account_id"]
            }
        }
    ]
}

name_to_account_id = {
    "Alice": "A123",
    "Bob": "B456",
    "Charlie": "C789"
}

account_id_to_bill = {
    "A123": 120.50,
    "B456": 200.75,
    "C789": 99.99
}

def get_account_id(name):
    return name_to_account_id.get(name, "Name not found")

def get_last_bill_amount(account_id):
    return account_id_to_bill.get(account_id, "Account ID not found")

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])

# Contributor's config - Please replace with your own, I have replaced mine with an Azure OpenAI endpoint.
config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4"],
    },
)

# create a prompt for our agent
billing_assistant_agent_prompt = '''
This agent is a helpful assistant that can retrieve the account id and the last bill amount for a customer. 
Any other customer care requests are outside the scope of this agent. 
Once you have completed assisting the user output TERMINATE'''
# create the agent and give it the config with our function definitions defined
billing_assistant_agent = autogen.AssistantAgent(
    name="billing_assistant_agent",
    system_message=billing_assistant_agent_prompt,
    llm_config=llm_config,
)

# first define our user proxy, remember to set human_input_mode to NEVER in order for it to
# properly execute functions
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)
# next we register our functions, this is pretty straight forward
# we simply provide a dictionary mapping the names we expect the llm to call to references to the actual functions
# it's good practice to have the names the llm tries to call be the same as the actual pre-defined functions
# but we can call them whatever we want,
user_proxy.register_function(
    function_map={
        "get_account_id": get_account_id,
        "get_last_bill_amount": get_last_bill_amount,
    }
)


class Config(object):
    MESSAGE = os.environ.get("MESSAGE")


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    # Initiates the chat with B2
    user_proxy.initiate_chat(billing_assistant_agent, message="My name is Bob, can you tell me what my last bill was?")
    return billing_assistant_agent.last_message().get('content')


@app.route("/settings")
def get_settings():
    return {"message": app.config["MESSAGE"]}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
