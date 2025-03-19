from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import traceback
import os
from dotenv import load_dotenv
import datetime


load_dotenv()


def query_formatter(query, custom_prompt: str= None):    
    llm_model = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        model_name=os.getenv("OPENAI_MODEL_NAME"),
    )

    # Get today's date and add it to every prompt
    today = datetime.datetime.now()
    current_date_info = f"\nCURRENT DATE: {today.strftime('%Y-%m-%d')}\nCURRENT TIMESTAMP: {int(today.timestamp())}\n"

    query_formatter_template_lowercase = """You are an assistant tasked with identifying model number and model category of product from given query. generate keywords base on extracted information.
        I have an query text that may containing a model type, model category. Your task is to:
        Translate it into english if the given query text is not in english.
        Identify and extract the model type, model category from the query.
        Generate lowercase keywords based on the extracted information in multiple formats, including but not limited to:
        space separated, strip-separated, underscore_separated, no separation
        Ensure the keywords are diverse but still relevant to the extracted information.
        Return the output in JSON format as follows:
        {{
            "keywords": [
                <keyword1>,
            ]
        }}
        Example Output:
        If the extracted information is:
        Model Type: LC-2030C NT
        Model Category: HPLC
        The JSON output should be:
        {{
            "keywords": [
                "lc",
                "2030c",
                "lc 2030c nt",
                "lc-2030c nt",
                "lc-2030c-nt",
                "lc2030cnt",
                "lc 2030cnt",
                "lc 2030c-nt",
                "hplc",
                "high performance liquid chromatography",
                "liquid chromatography"
            ],
        }}
        Make sure the keywords are properly formatted and case-sensitive to match potential search queries. Make sure only return json"""
    
    text_q = f"identify this query : {query}"
    if custom_prompt:
        query_formatter_template_lowercase = custom_prompt
        text_q = f"{query}"

    # Add current date information to every prompt
    if custom_prompt:
        query_formatter_template_lowercase = custom_prompt + current_date_info
    else:
        query_formatter_template_lowercase = query_formatter_template_lowercase + current_date_info

    prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": query_formatter_template_lowercase},
            {"role": "user", "content": [
                {"type": "text", "text": text_q}
            ]}
        ])
    
    query_formatter_chain = (
        {"query": lambda _: query}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    
    try:
        print(f"Using prompt with current date: {today.strftime('%Y-%m-%d')}")
        response = query_formatter_chain.invoke({})
        return response
    except Exception as e:
        print("WARNING: failed to format query, returning empty keywords")
        print(traceback.format_exc())
        return """{"keywords": []}"""