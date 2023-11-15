import asyncio

from langchain.prompts import PromptTemplate
import openai
import streamlit as st


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


loop = get_or_create_eventloop()
asyncio.set_event_loop(loop)

from langchain.chains import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.context_var_chain import ContextVarChain

from constants import COLANG_CONFIG, openai_api_key, template, YAML_CONFIG


def get_model(model_name):
    config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)

    app = LLMRails(config)

    openai.api_key = openai_api_key

    constitutional_chain = ConstitutionalChain.from_llm(
        llm=app.llm,
        chain=ContextVarChain(var_name="last_bot_message"),
        constitutional_principles=[
            ConstitutionalPrinciple(
                critique_request="Tell if this answer is good.",
                revision_request="Give a better answer.",
            )
        ],
    )
    app.register_action(constitutional_chain, name="check_if_constitutional")

    return app


st.title("📈 Personal Financial Advisor")
st.caption("A chatbot powered by OpenAI LLMs")

model = get_model("gpt-3.5-turbo")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        content = msg["content"].split("\nQuestion: ")[-1]
    else:
        content = msg["content"]
    st.chat_message(msg["role"]).write(content)

if prompt := st.chat_input():
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    prompt_query = prompt_template.format(question=prompt)
    st.session_state.messages.append({"role": "user", "content": prompt_query})
    st.chat_message("user").write(prompt)
    response = model.generate(messages=st.session_state.messages)
    msg = response.get("content")
    st.session_state.messages.append(response)
    st.chat_message("assistant").write(response["content"])
