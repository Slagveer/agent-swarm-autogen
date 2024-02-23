import datetime
import os
import subprocess
import time
import wget

import autogen
import wget
from flask import Flask
from langchain.chains import LLMChain, SimpleSequentialChain, TransformChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import requests
from PyPDF2 import PdfReader
import io

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

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


# Open the text file in read mode ('r')
with open('2680-0.txt', 'r', encoding='utf-8') as file:
    # Read the entire content of the file
    content = file.read()
    # Iterate over each line in the file
    for line in file:
        # Print the current line
        print(line.strip())


def transform_func(inputs: dict) -> dict:
    """
    Extracts specific sections from a given text based on newline separators.

    The function assumes the input text is divided into sections or paragraphs separated
    by one newline characters (`\n`). It extracts the sections from index 922 to 950
    (inclusive) and returns them in a dictionary.

    Parameters:
    - inputs (dict): A dictionary containing the key "text" with the input text as its value.

    Returns:
    - dict: A dictionary containing the key "output_text" with the extracted sections as its value.
    """
    print(inputs)
    text = inputs["text"]
    shortened_text = "\n".join(text.split("\n")[921:950])
    return {"output_text": shortened_text}

transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_func, verbose=True
)

class Config(object):
    MESSAGE = os.environ.get("MESSAGE")


app = Flask(__name__)
app.config.from_object(Config)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    t = time.localtime()
    # running chain
    review: str = transform_chain.run(content)
    return review + str(time.strftime("%H:%M:%S", t))


@app.route("/settings")
def get_settings():
    return {"message": app.config["MESSAGE"]}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
