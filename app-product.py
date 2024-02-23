import datetime
import os
import time

import autogen
from flask import Flask
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

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

# Step 1: Generate a product name
product_name_prompt = PromptTemplate(
    input_variables=["product_category"],
    template="Generate a product name for a {product_category} product.",
)
product_name_chain = LLMChain(llm=llm, prompt=product_name_prompt)

# Step 2: Generate a company name
company_name_prompt = PromptTemplate(
    input_variables=["product_name"],
    template="Create a company name that specializes in {product_name}.",
)
company_name_chain = LLMChain(llm=llm, prompt=company_name_prompt)

# Step 3: Design a logo tagline
logo_tagline_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Design a tagline for the logo of {company_name}.",
)
logo_tagline_chain = LLMChain(llm=llm, prompt=logo_tagline_prompt)

# Step 4: Come up with a marketing slogan
marketings_slogan_prompt = PromptTemplate(
    input_variables=["product_name"],
    template="Come up with a marketing slogan for a company that sells {product_name}.",
)
marketings_slogan_chain = LLMChain(llm=llm, prompt=marketings_slogan_prompt)

# Combining the chains
overall_chain = SimpleSequentialChain(
    chains=[
        product_name_chain, company_name_chain, logo_tagline_chain, marketings_slogan_chain
    ],
    verbose=True,
)


class Config(object):
    MESSAGE = os.environ.get("MESSAGE")


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    t = time.localtime()
    # running chain
    result = overall_chain.run(["car", product_name_chain])
    return result + str(time.strftime("%H:%M:%S", t))


@app.route("/settings")
def get_settings():
    return {"message": app.config["MESSAGE"]}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
