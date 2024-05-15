import streamlit as st
st.title("🤖 English to SQL - LangChain project")
st.subheader("This is sqlGPT, your data analytics assistant.")#, divider='gray')


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_core.runnables.history import RunnableWithMessageHistory
import RAG_txt2sql as rag_arch

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    st.chat_message("user").write(prompt)
    
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_arch.rag_chain,
    rag_arch.get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": "abc123"}},
        )["answer"]
    # response = db_chain.run(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
