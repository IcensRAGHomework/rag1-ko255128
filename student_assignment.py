import base64
import json
import traceback
from http.client import responses
from mimetypes import guess_type

import requests
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import StructuredTool
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
api_key = "8CoOjf5fS1nMOkPAomVIgGcv5KO94Dil"
llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

class GetHolidaySchema(BaseModel):
    year: int = Field(description="the holiday of year")
    month: int = Field(description="the holiday of month")

def get_holidays(year, month):
    url = f"https://calendarific.com/api/v2/holidays?&api_key={api_key}&country=tw&year={year}&month={month}"
    response = requests.get(url)
    response = response.json()
    response = response.get('response')
    return response


def generate_hw01(question):
    responses_schema = [
        ResponseSchema(
            name="date",
            description="紀念日或節日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="紀念日或節日的中文名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=responses_schema)
    format_instruction = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "使用台灣繁體中文並回答問題, {format_instruction}"),
        ("human", "{question}, 並將結果再用Result包起來")])
    prompt = prompt.partial(format_instruction=format_instruction)
    responses = llm.invoke(prompt.format_messages(question=question)).content
    responses = responses.replace("```","")
    responses = responses.replace("json\n", "")
    return responses

    
def generate_hw02(question):
    responses_schema = [
        ResponseSchema(
            name="date",
            description="紀念日或節日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="紀念日或節日的中文名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=responses_schema)
    format_instruction = output_parser.get_format_instructions()
    tool = StructuredTool.from_function(
        name="get_holidays",
        description="year年台灣month月紀念日有哪些?",
        func=get_holidays,
        args_schema=GetHolidaySchema,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. 並請您使用台灣繁體中文並回答問題, {format_instruction}, 請將結果回在同一個Jason格式內並塞到Result之下"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")])
    prompt = prompt.partial(format_instruction=format_instruction)
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"input": question}).get('output')
    response = response.replace("\n```", "")
    response = response.replace("```json\n", "")
    return response


def generate_hw03(question2, question3):
    responses_schema = [
        ResponseSchema(
            name="date",
            description="紀念日或節日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="紀念日或節日的中文名稱")
    ]
    responses_schema2 = [
        ResponseSchema(
            name="add",
            description="這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。"),
        ResponseSchema(
            name="reason",
            description="描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。")
    ]
    output_parser = StructuredOutputParser(response_schemas=responses_schema)
    output_parser2 = StructuredOutputParser(response_schemas=responses_schema2)
    format_instruction = output_parser.get_format_instructions()
    format_instruction2 = output_parser2.get_format_instructions()
    tool = StructuredTool.from_function(
        name="get_holidays",
        description="year年台灣month月紀念日有哪些?",
        func=get_holidays,
        args_schema=GetHolidaySchema,
    )
    history = ChatMessageHistory()
    def get_history() -> ChatMessageHistory:
        return history
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. 請您使用台灣繁體中文並回答問題, {format_instruction}, 請將結果回在同一個Jason格式內並塞到Result之下"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")])
    prompt = prompt.partial(format_instruction=format_instruction)
    tools = [tool]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    q2response = agent_with_chat_history.invoke({"input": question2}).get('output')
    print("Q2 Ans:", q2response)
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "請您使用台灣繁體中文並回答問題, {format_instruction}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")])
    prompt2 = prompt2.partial(format_instruction=format_instruction2)
    runnable = prompt2 | llm
    chat_with_history = RunnableWithMessageHistory(
        runnable,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    q3response = chat_with_history.invoke({"input": question3}).content
    q3response = q3response.replace("\n```", "")
    q3response = q3response.replace("```json\n", "")
    dict = json.loads(q3response)
    dict = {"Result": dict}
    q3response = json.dumps(dict, indent = 4, ensure_ascii=False)
    return q3response
    
def generate_hw04(question):
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)
    responses_schema = [
        ResponseSchema(
            name="score",
            description="積分")
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "辨識圖片中的文字表格, {format_instruction}"),
            ("user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                ],
            ),
            ("human", "{question}")
        ]
    )
    output_parser = StructuredOutputParser(response_schemas=responses_schema)
    format_instruction = output_parser.get_format_instructions()
    prompt = prompt.partial(format_instruction=format_instruction)
    response = llm.invoke(prompt.format_messages(question=question)).content
    response = response.replace("\n```", "")
    response = response.replace("```json\n", "")
    dict = json.loads(response)
    dict = {"Result": dict}
    response = json.dumps(dict, indent = 4, ensure_ascii=False)
    return response

    
def demo(question):

    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

if __name__ == '__main__':
    # responses = generate_hw02("2024年台灣10月紀念日有哪些?")
    # responses = generate_hw03("2024年台灣10月紀念日有哪些?",
    #                           "根據先前的節日清單，這個節日{\"date\": \"10-31\", \"name\": \"蔣公誕辰紀念日\"}是否有在該月份清單？")
    responses =  generate_hw04("中華台北的積分")
    print(responses)
