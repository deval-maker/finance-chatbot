import os

openai_api_key = os.environ.get("OPENAI_API_KEY")

template_finance = """As an experienced personal financial advisor, generate response the following question. \
        If the question is not related to finance or personal finance, please respond with \
        'I am sorry, I am not able to answer this question.' \nQuestion: {question}"""  # noqa: E501


COLANG_CONFIG = open("config/colang_config.co").read()
YAML_CONFIG = open("config/llm_config.yaml").read()

template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """
