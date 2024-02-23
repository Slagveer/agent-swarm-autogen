import datetime
import os
import time

import autogen
from flask import Flask
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain


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

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])

os.environ['OPENAI_API_KEY'] = "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"

# creating a general LLM to be used by other chains
llm = OpenAI(openai_api_key="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")

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
]

destination_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")


class Config(object):
    MESSAGE = os.environ.get("MESSAGE")


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    t = time.localtime()
    # running chain
    review: str = default_chain.invoke("What is HipHop?")
    return review


@app.route("/settings")
def get_settings():
    return {"message": app.config["MESSAGE"]}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
