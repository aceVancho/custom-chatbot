import sys
import os
import requests

import gradio as gr
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from llama_index import download_loader, GPTSimpleVectorIndex
from langchain import OpenAI
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
gpt_model_name = "gpt-4"
def construct_index_from_docs(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-4", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

def construct_index_from_website(url):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name=gpt_model_name, max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

    loader = BeautifulSoupWebReader()

    documents = loader.load_data(urls=[url])

    index = GPTSimpleVectorIndex.from_documents(documents)

    index.save_to_disk('index.json')

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

url = input('Enter a URL you would like to learn about:  ')

user_textbox = gr.inputs.Textbox(
    lines=20, 
    label="Enter your text")

chatbot_textbox = gr.inputs.Textbox(
    lines=20, 
    label="Response:",)

iface = gr.Interface(
    fn=chatbot,
    inputs=user_textbox,
    outputs=chatbot_textbox,
    title=f"Chatbot based on {url}",
    description=f"This chatbot runs on ChatGPT's {gpt_model_name} model"
    )

# Pick one
# index = construct_index_from_docs('docs')
index = construct_index_from_website(url)

iface.launch(share=True)