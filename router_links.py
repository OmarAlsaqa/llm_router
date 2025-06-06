from typing import List, Dict, Any, Union
import os
import grpc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
import asyncio
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import re
from langchain_community.document_transformers import Html2TextTransformer
from langchain.docstore.document import Document

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="grpc")

# Initialize Gemini LLM
os.environ["GOOGLE_API_KEY"] = "***"
os.environ["JINA_API_KEY"] = "***"
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", google_api_key=api_key, temperature=0.0
)


# --- 1. Define the State ---
class GraphState(BaseModel):
    """Represents the state of our graph."""

    question: str = None
    route: str = None
    response: Union[str, dict] = None


# --- 2. Define Sub-Committee Tools and Models ---


# Sub-Committee 1: Assistant
async def assistant_agent(state: GraphState):
    print("Starting assistant_agent")
    template = """You are a helpful assistant.
  User: {question}
  Assistant: """
    prompt = ChatPromptTemplate.from_template(template)
    prompt_value = await prompt.ainvoke({"question": state.question})
    messages = prompt_value.to_messages()
    formatted_messages = [
        {"role": m.role if hasattr(m, "role") else "user", "content": m.content}
        for m in messages
    ]
    response = await llm.ainvoke(formatted_messages)
    print("Assistant agent response: ", response)
    return GraphState(question=state.question, response=response.content)


# Sub-Committee 2: Web Search
async def web_search_agent(state: GraphState):
    print("Starting web_search_agent")
    search = DuckDuckGoSearchRun()
    search_results = search.run(state.question)

    template = """You are a helpful assistant, tasked with providing information based on web search results.
    User question: {question}
    Web search results: {search_results}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    prompt_value = await prompt.ainvoke(
        {"question": state.question, "search_results": search_results}
    )
    messages = prompt_value.to_messages()
    formatted_messages = [
        {"role": m.role if hasattr(m, "role") else "user", "content": m.content}
        for m in messages
    ]
    response = await llm.ainvoke(formatted_messages)
    print("Web search agent response: ", response)
    return GraphState(question=state.question, response=response.content)


# Sub-Committee 3: Analyzer (Modified)
async def analyzer_agent(state: GraphState):
    print("Starting analyzer_agent")
    url_match = re.search(r"https?://[^\s]+", state.question)
    if not url_match:
        return GraphState(question=state.question, response="No URL found in the question.")

    url = url_match.group(0)

    try:
        # Fetch markdown content using Jina AI
        full_url = f"https://r.jina.ai/{url}"
        headers = {
            "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",
            "Content-Type": "application/json",
            "X-Return-Format": "markdown"
        }

        response = requests.get(full_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        markdown_content = response.text

        print("Markdown content: ", markdown_content)

        # Enhance markdown content by reintegrating URLs using BeautifulSoup
        soup = BeautifulSoup(requests.get(url).text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            a_tag["href"] = urljoin(url, a_tag["href"])

        for link in soup.find_all("a", href=True):
            if link.get("href") not in markdown_content:
                markdown_content += f"\n[Link]({link.get('href')})"

        # Prepare LLM prompt for metadata extraction
        template = """You are a helpful assistant skilled in analyzing web page content. 
        The following is the markdown version of a webpage. Your task is to extract metadata for any document files listed on the page.
        Metadata includes:
        - File name or description
        - URL
        - Date (if available)
        - Type (if available)
        - Size (if available)

        Use the markdown content below to extract this information. Provide the output as a JSON array of objects where each object represents a document with its metadata.

        Markdown content:
        {markdown_content}

        Extracted metadata:"""

        prompt = ChatPromptTemplate.from_template(template)
        prompt_value = await prompt.ainvoke({"markdown_content": markdown_content})
        messages = prompt_value.to_messages()
        formatted_messages = [
            {"role": m.role if hasattr(m, "role") else "user", "content": m.content}
            for m in messages
        ]

        # Send prompt to the LLM
        response = await llm.ainvoke(formatted_messages)

        # Parse and return the LLM output
        return GraphState(
            question=state.question,
            response=response.content
        )

    except requests.exceptions.RequestException as e:
        return GraphState(question=state.question, response=f"Error fetching URL: {e}")




# Sub-Committee 4: Summarizer
async def summarizer_agent(state: GraphState):
    print("Starting summarizer_agent")
    template = """You are a helpful summarizer.
    User question: {question}
    Summary: """

    prompt = ChatPromptTemplate.from_template(template)
    prompt_value = await prompt.ainvoke({"question": state.question})
    messages = prompt_value.to_messages()
    formatted_messages = [
        {"role": m.role if hasattr(m, "role") else "user", "content": m.content}
        for m in messages
    ]
    response = await llm.ainvoke(formatted_messages)
    print("Summarizer agent response: ", response)
    return GraphState(question=state.question, response=response.content)


# --- 3. Define Router ---
class RouteOutput(BaseModel):
    """Route output."""

    route: str = Field(description="Route to take")


router_parser = PydanticOutputParser(pydantic_object=RouteOutput)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at routing user questions to the correct sub-committee.
        The following sub-committees are available:
        - assistant: For general questions that don't require any specialized tool.
        - web_search: For questions requiring up-to-date information from the web.
        - analyzer: For questions involving analysis of data or text, particularly listing files from URL with all the metadata available.
        - summarizer: For questions focused on summarizing text.
        
        Given the user question, identify which of the sub-committees would be best suited to answer it.
        
        Format your response as a JSON object with a single key: "route" followed by the name of the sub-committee.
        {format_instructions}
        """,
        ),
        ("user", "{question}"),
    ]
)


async def router(state: GraphState):
    print("Starting Router")
    prompt_value = await router_prompt.ainvoke(
        {
            "question": state.question,
            "format_instructions": router_parser.get_format_instructions(),
        }
    )
    messages = prompt_value.to_messages()
    formatted_messages = [
        {"role": m.role if hasattr(m, "role") else "user", "content": m.content}
        for m in messages
    ]
    response = await llm.ainvoke(formatted_messages)
    parsed_response = router_parser.parse(response.content)
    print("Router response: ", parsed_response.route)
    return GraphState(question=state.question, route=parsed_response.route)


# --- 4. Define Graph ---
def route_check(state):
    if state.route == "assistant":
        return "assistant"
    elif state.route == "web_search":
        return "web_search"
    elif state.route == "analyzer":
        return "analyzer"
    elif state.route == "summarizer":
        return "summarizer"
    else:
        return "end"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("router", router)
workflow.add_node("assistant", assistant_agent)
workflow.add_node("web_search", web_search_agent)
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("summarizer", summarizer_agent)

# Set up the edges
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_check,
    {
        "assistant": "assistant",
        "web_search": "web_search",
        "analyzer": "analyzer",
        "summarizer": "summarizer",
        "end": END,
    },
)


workflow.add_edge("assistant", END)
workflow.add_edge("web_search", END)
workflow.add_edge("analyzer", END)
workflow.add_edge("summarizer", END)

# Compile the graph
chain = workflow.compile()


# --- 5. Execution Function ---
async def run_pipeline(question):
    inputs = GraphState(question=question)
    result = await chain.ainvoke(inputs)
    print("Final result: ", result)
    return result.get("response", {})


# --- 6. Example Usage ---
async def main():
    try:
        while True:
            question = input("Enter your question (or 'exit' to quit): ")
            if question.lower() == "exit":
                print("Shutting down...")
                break
            result = await run_pipeline(question)
            print(f"\n{result}\n")
    finally:
        # Explicitly clean up gRPC resources
        await grpc.aio._shutdown_all()  # Shutdown all gRPC async resources
        await asyncio.sleep(0.5)  # Allow grpc-aio to clean up properly


if __name__ == "__main__":
    asyncio.run(main())
