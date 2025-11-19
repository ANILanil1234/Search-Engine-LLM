from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain.agents import create_agent
import os
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq



wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

dudu=DuckDuckGoSearchRun(name="dudu")

tools=[wiki,arxiv,dudu]

st.title("Welcome to Chatbot Without messagae History using Mistral")
st.sidebar.header("Model Details")

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Enter your Query"}]
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

api_key = st.sidebar.text_input("Enter your API key",type="password",placeholder="Enter your API key")

if api_key:
    model = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.7
    )
    agent = create_agent(
        model=model,
        tools=tools,
    )
    st.sidebar.write("Model is Ready")

question=st.chat_input("Enter prompt")
if not question:
    st.warning("Please enter your question")


if question and api_key:
    st.session_state.messages.append({"role":"user","content":question})
    st.chat_message('user').write(question)
    with st.spinner("🔄 Searching..."):
            resp = agent.invoke({"messages": [{"role": "user", "content": question}]})
            final_response = resp["messages"][-1].content

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": final_response})

            # Display result
            st.chat_message("assistant").write(final_response)
            st.success("✅ Search Complete!")


