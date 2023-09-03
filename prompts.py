DEFAULT_REFINE_PROMPT = """You are the AI lawyer bot. the following is the conversation between you and human. genarate a question followed by '?' as helpful answer to gather additional information from human:

Examples:

Message: Hello?
Reply: 
----------------

Message: {message}
Chit-chat:
"""

DEFAULT_QA_PROMPT = """Use the following pieces of chat history and context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
------------------
{chat_history}

------------------
{context}
Helpful Answer:"""