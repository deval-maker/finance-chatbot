from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import os 
from constants import template
from langchain.chat_models import ChatOpenAI


def get_llm_chain_and_memory():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    llm_chain = LLMChain(llm=ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-0613", temperature=0), prompt=prompt, memory=memory)
    return llm_chain, msgs
