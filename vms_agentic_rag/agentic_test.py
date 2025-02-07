import asyncio
import vms_agentic_rag as vms
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Initialize the Chatbot with the directory containing PDF files and other parameters
    chatbot = vms.VMSAgenticRag(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        prompt=""
    )

    # prompt = """i want to see my schedule"""
    # prompt = """get list activities"""
    # prompt = """get list of activities to cancel"""
    prompt = "3"
    prompt = "Cancel no 1"
    response = await chatbot.get_response(input_prompt=prompt, phone_number="1234567890")
    
    # Print the response
    print("Chatbot response:", response)

# Run the test
if __name__ == "__main__":
    asyncio.run(main())
