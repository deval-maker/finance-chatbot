import os

from langchain.prompts import PromptTemplate
import openai
import streamlit as st


template = """As an experienced personal financial advisor, generate response the following question. \
        If the question is not related to finance or personal finance, please respond with \
        'I am sorry, I am not able to answer this question.' \nQuestion: {question}"""  # noqa: E501

openai_api_key = os.environ.get("OPENAI_API_KEY")

st.title("📈 Personal Financial Advisor")
st.caption("A chatbot powered by OpenAI LLMs")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        content = msg["content"].split("\nQuestion: ")[-1]
    else:
        content = msg["content"]
    st.chat_message(msg["role"]).write(content)

if prompt := st.chat_input():
    openai.api_key = openai_api_key
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    prompt_query = prompt_template.format(question=prompt)
    st.session_state.messages.append({"role": "user", "content": prompt_query})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=st.session_state.messages
    )
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
