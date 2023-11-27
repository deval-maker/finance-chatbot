import streamlit as st

from llm_agent import get_retreiver_chain


st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

llm_chain, msgs = get_retreiver_chain()

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you? I am well versed in the topic of Anyscale-Ray, and a few products from Nvidia Triton, NeMo, and NeMo Guardrails.")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    msgs.add_user_message(prompt)
    st.chat_message("human").write(prompt)
    response = llm_chain({"chat_history": msgs.messages, "question": prompt})
    msgs.add_ai_message(response["answer"])
    st.chat_message("ai").write(response["answer"])
