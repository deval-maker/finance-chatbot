import streamlit as st
from llm_agent import get_llm_chain_and_memory

st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

llm_chain, msgs = get_llm_chain_and_memory()

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)
