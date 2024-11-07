from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from pymessenger import Bot
import os
from typing import TypedDict, Sequence
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Add near the top of the file with other imports
def load_system_prompt():
    with open("system_prompt.txt", "r") as file:
        return file.read().strip()

def load_knowledge_base():
    system_prompt = load_system_prompt()
    try:
        with open("devobi_faq.txt", "r") as file:
            faq_content = file.read().strip()
        return f"{system_prompt}\n\nKnowledge Base:\n{faq_content}"
    except FileNotFoundError:
        return system_prompt

# Define the FastAPI app
app = FastAPI()
messenger = Bot(os.getenv("PAGE_ACCESS_TOKEN"))

def get_weather(location: str) -> str:
    return f"Weather information for {location}"

def search_products(query: str) -> str:
    return f"Product results for {query}"

tools = [
    StructuredTool.from_function(func=get_weather, name="get_weather", description="Get weather information for a specific location"),
    StructuredTool.from_function(func=search_products, name="search_products", description="Search for products in the database")
]

llm = ChatOpenAI(temperature=0.2, model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next: str

workflow = StateGraph(AgentState)

def message_handler(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages:
        return {"messages": messages, "next": END}
     
    latest_message = messages[-1]
    if isinstance(latest_message, HumanMessage):
        response = llm.invoke(messages)
        return {"messages": list(messages) + [response], "next": "use_tools"}
    return {"messages": messages, "next": END}

def tool_handler(state: AgentState) -> AgentState:
    messages = state["messages"]
    for tool in tools:
        if tool.name in messages[-1].content:
            result = tool.func(messages[-1].content)
            return {"messages": list(messages) + [AIMessage(content=result)], "next": END}
    return {"messages": messages, "next": END}

workflow.add_node("process_message", message_handler)
workflow.add_node("use_tools", tool_handler)
workflow.add_edge("process_message", "use_tools")
workflow.add_edge("use_tools", END)
workflow.set_entry_point("process_message")
chain = workflow.compile()

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = dict(request.query_params)
    if params.get("hub.mode") and params.get("hub.verify_token"):
        if params["hub.mode"] == "subscribe" and params["hub.verify_token"] == os.getenv("VERIFY_TOKEN"):
            return int(params["hub.challenge"])
    return {"error": "Invalid request"}

@app.post("/webhook")
async def handle_message(request: Request):
    data = await request.json()
    if data["object"] == "page":
        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                if messaging_event.get("message", {}).get("text"):
                    system_message = SystemMessage(content=load_knowledge_base())
                    
                    initial_state = {
                        "messages": [
                            system_message,
                            HumanMessage(content=messaging_event["message"]["text"])
                        ],
                        "next": ""
                    }
                    result = chain.invoke(initial_state)
                    messenger.send_text_message(messaging_event["sender"]["id"], result["messages"][-1].content)
    return {"success": True}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
