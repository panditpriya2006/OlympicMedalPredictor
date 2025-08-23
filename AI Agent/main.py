from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor # how we execute the agent
from chatBot.tools import search_tool,wiki_tool, save_tool
load_dotenv() #loads env variable file so we have credentials



#now we are gonna do prompt template
#defint python class saying what we want it to generate 
#then going to format output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools__used:list[str]
llm = ChatAnthropic(model = "claude-3-4-sonnet-20241022")
#test if its loaded
#response = llm.invoke("What is the meaning of life?")
#print(response)
parser = PydanticOutputParser(pydantic_object= ResearchResponse)
#using pydantic schema

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", #info to the LLM so it knows what it is doing
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) #partially gonna fill in this prompt by passing in format instructs that we created before
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt = prompt,
    tools = tools
)
agent_executor = AgentExecutor(agent=agent, tools =[], verbose =True) #verbose is if u wanna see what the agent thought process
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)
try:
    structure_response = parser.parse(raw_response.get("output")[0]["text"])#only structure the output
#print(structure_response.topic) #access the topic(because is an object) #->python object now
except Exception as e:
    print("Error parsing response",e,"Raw Response - ", raw_response )