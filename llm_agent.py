from operator import itemgetter
import os
from typing import Sequence

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnableMap
from langchain.vectorstores import Weaviate
import weaviate
from langchain.chains import ConversationalRetrievalChain

from constants import (
    REPHRASE_TEMPLATE,
    RESPONSE_TEMPLATE,
    template,
    WEAVIATE_DOCS_INDEX_NAME,
    WEAVIATE_URL,
)


def get_llm_chain_and_memory():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    llm_chain = LLMChain(
        llm=ChatOpenAI(
            openai_api_key=openai_api_key, model="gpt-3.5-turbo-0613", temperature=0, streaming=True
        ),
        prompt=prompt,
        memory=memory,
    )
    return llm_chain, msgs


def get_retreiver_chain():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    retriever = get_retriever()
    model = ChatOpenAI(model='gpt-3.5-turbo-16k') # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    return qa, msgs


def get_embeddings_model():
    return OpenAIEmbeddings(chunk_size=200)


def get_retriever() -> BaseRetriever:
    weaviate_client = weaviate.Client(url=WEAVIATE_URL)
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))


def create_retriever_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(run_name="Itemgetter:question")
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | response_synthesizer
    )
