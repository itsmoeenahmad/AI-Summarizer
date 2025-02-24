# Importing Packages
import streamlit as streamlit
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
import os

# Loading LLM API_KEY
load_dotenv()


# LLM
llm = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature": 0.0, "max_length": 300},
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Prompt Template
promptTemplate = PromptTemplate(
    input_variables=['text'],
    template=(
       "Summarize the following text into exactly one paragraph. "
        "Do not suggest improvements, alternatives, formatting options—only or options return a summary.\n\n"
        "Text: {text}\n\n"
    )
)


# Streamlit - Title
streamlit.title('AI Summarizer')

# Streamlit - Mini Text
streamlit.text('~ By Moeen Ahmad')

# Taking User Imput
userInput = streamlit.text_area('Enter or Paste Your Text', height=200)

# Making a Prompt
prompt = promptTemplate.format(text=userInput)

# Streamlit - Button
buttonClicked = streamlit.button('Summarize')


if buttonClicked:
    if not userInput.strip():
        streamlit.warning('Enter or Type Text Before Clicking On Button')
    elif len(userInput.strip()) < 50: 
        streamlit.warning('Please enter at least 50 characters to summarize.')
    else:
        with streamlit.spinner('Summarizing...'):
            response = llm.invoke(prompt)  
            streamlit.write(response)



