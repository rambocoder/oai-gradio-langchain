# from langchain.llms.fake import FakeListLLM
from langchain_openai import ChatOpenAI
# from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

template = """Given this text, decide what is the issue the customer is
concerned about. Valid categories are these:
* product issues
* delivery problems
* missing or late orders
* wrong product
* cancellation request
* refund or exchange
* bad support experience
* no clear reason to be upset
Text: {email}
Category:
"""
prompt = PromptTemplate(template=template, input_variables=["email"])
# tools = load_tools(["Python_REPL"])
responses = ["Action: Python_REPL\nAction Input: print(2 +2)", "Final Answer: 4"]
# llm = FakeListLLM(responses = responses)


llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

# print(llm("Hello world"))

customer_email = "Your product is awesome"

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
print(llm_chain.run(customer_email))


# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )
# agent.run("whats 2+2")
