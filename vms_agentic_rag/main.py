import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vms_agentic_rag import VMSAgenticRag
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class RequestBody(BaseModel):
    phone_number: str
    input_prompt: str

# Initialize the Chatbot
chatbot = VMSAgenticRag(
    model_name=os.getenv("OPENAI_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    prompt=""
)

@app.post("/get_response")
async def get_response(request_body: RequestBody):
    try:
        response = await chatbot.get_response(
            input_prompt=request_body.input_prompt,
            phone_number=request_body.phone_number
        )

        chat_history = chatbot.get_chat_history(phone_number=request_body.phone_number)
        print("================================================================")
        print("Chat history:", chat_history)
        print("================================================================")

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
