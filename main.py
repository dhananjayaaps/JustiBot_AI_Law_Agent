from fastapi import FastAPI
from pydantic import BaseModel
from llm import chat

app = FastAPI()

class Message(BaseModel):
    message: str

@app.post("/chat")
async def send_message(data: Message, response_class=None, response_content_type="text/plain"):
    res = chat(data.message)
    return str(res)