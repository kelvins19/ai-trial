import asyncio
from chatbot_1 import Chatbot1
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Initialize the Chatbot with the directory containing PDF files and other parameters
    chatbot = Chatbot1(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        prompt="",
        pdf_dir="../assets/"
    )

    # Test the _call_openrouter_batch method with a sample prompt
    # prompt = "What is LC-MS"
    # prompt = "What is the dimensions of GC-2050"
    prompt = """Return the Simultaneous Determination of Five Genotoxic 
Aryl Sulfonate Impurities in Pharmaceuticals by 
LCMS-2050 analysis condition table"""
    response = await chatbot._call_openrouter_batch(prompt)
    
    # Print the response
    print("Chatbot response:", response)

# Run the test
if __name__ == "__main__":
    asyncio.run(main())
