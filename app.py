pip install streamlit
pip install google.generativeai
pip install langchain_community
pip install langchain
pip install pypdf
pip install docarray
pip install chromadb
pip install google-cloud-aiplatform
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatGooglePalm
from langchain.document_loaders import PyPDFLoader
import os

# Function to handle note-taking
def take_notes(question, answer, source_documents):
    note = {
        "question": question,
        "answer": answer,
        "source_documents": [doc.page_content for doc in source_documents] if source_documents else []
    }
    return note

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = GooglePalmEmbeddings(google_api_key='AIzaSyB5xBomrmwEg5Eu4ZUxjuv6nwvxDamG5yE')
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

def main():
    st.title("Digital Library Assistant")

    # Load PDF file
    file = st.text_input("Please enter the path to your PDF file:")
    k = st.number_input("Enter the number of similar documents to retrieve (k):", min_value=1, step=1, value=3)

    if st.button("Load Database"):
        qa_chain = load_db(file, "stuff", k)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retriever = qa_chain.retriever
        llm = ChatGooglePalm(google_api_key=os.environ['API_KEY'])

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        st.write("Database loaded successfully. You can start asking questions now.")

        # Store notes
        notes = []

        while True:
            user_input = st.text_input("You:")
            if st.button("Submit"):
                if user_input.lower() in ["exit", "quit"]:
                    st.write("Goodbye! Have a great day!")
                    break

                result = qa({"question": user_input})

                # Debugging print to understand the structure of result
                st.write("Debug - Full Result:", result)

                if "answer" in result:
                    answer = result["answer"]
                    st.write("Assistant:", answer)

                    # Store the result as a note
                    note = take_notes(user_input, answer, result.get("source_documents", []))
                    notes.append(note)
                    with open('research_notes.txt', 'w') as file:
                        for note in notes:
                            file.write(f"Question: {note['question']}\n")
                            file.write(f"Answer: {note['answer']}\n")
                            for doc in note["source_documents"]:
                                file.write(f"Source Document: {doc}\n")
                            file.write("\n")
                else:
                    st.write("Unexpected result format. Please check the debug output.")

if __name__ == "__main__":
    main()
