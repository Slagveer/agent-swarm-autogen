import datetime
import os
import time

import autogen
from flask import Flask
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory

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

# rapper
rapper_template: str = """You are an American rapper, your job is to come up with\
lyrics based on a given topic

Here is the topic you have been asked to generate a lyrics on:
{input}\
"""

rapper_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["input"], template=rapper_template)

# creating the rapper chain
rapper_chain: LLMChain = LLMChain(llm=llm, output_key="lyric", prompt=rapper_prompt_template)

# verifier
verifier_template: str = """You are a verifier of rap songs, you are tasked\
to inspect the lyrics of rap songs. If they consist of violence and abusive languge\
flag the lyrics. 

Your response should be in the following format, as a Python Dictionary.
lyric: this should be the lyric you received 
Violence_words: True or False

Here is the lyrics submitted to you:
{lyric}\
"""

verified_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["lyric"], template=verifier_template)

# creating the verifier chain
verifier_chain: LLMChain = LLMChain(
    llm=llm, output_key="AI_verified", prompt=verified_prompt_template)

# final output chain
final_template: str = """You are a final quality assurance of a lyrics post.\
Your job will be to accept a lyric and output data in the following format

Your final response should be in the following format, in a Python Dictionary format:
lyric: this should be the lyric you received
Date and time verified: {time_created_and_verified}
Verified by human: {verified_by_human}

Here is the lyric submitted to you:
{AI_verified}\
"""

final_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["AI_verified", "time_created_and_verified", "verified_by_human"], template=final_template)

# creating the verifier chain
final_chain: LLMChain = LLMChain(
    llm=llm, output_key="final_output", prompt=final_prompt_template)


# creating the simple sequential chain
timer = time.localtime()
ss_chain: SequentialChain = SequentialChain(
    memory=SimpleMemory(memories={
        "time_created_and_verified": str(time.strftime("%H:%M:%S", timer)), "verified_by_human": "False"}),
    chains=[rapper_chain, verifier_chain, final_chain],
    # multiple variables
    input_variables=["input"],
    output_variables=["final_output"],
    verbose=True)

class Config(object):
    MESSAGE = os.environ.get("MESSAGE")


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    t = time.localtime()
    # running chain
    review: str = ss_chain.invoke("Gun violence")
    return review


@app.route("/settings")
def get_settings():
    return {"message": app.config["MESSAGE"]}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
