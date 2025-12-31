from langchain_groq import ChatGroq
from  secret_key import GROQ_API_KEY
import os
import streamlit as st
import time
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.globals import set_debug


# Setup API Key from Colab Secrets
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 3. Initialize the Free Model (Llama 3)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6
)

# Initialize embeddings (used for both creating and loading FAISS index)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("News Research Tool 📈📈📈")
st.sidebar.title("News Article URls")

urls=[]
for i in range(3):
  url = st.sidebar.text_input(f"URL {i+1}")
  urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "my_faiss_index"  # Directory path, not file path
main_placeholder = st.empty()

if process_url_clicked:
  loader = UnstructuredURLLoader(urls=urls)
  main_placeholder.text("Data Loading...Started....✅✅✅")
  data = loader.load()
  recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' '],      # The character to split on
    chunk_size=1000,       # The maximum number of characters in each chunk
    chunk_overlap=200,     # How much text to repeat between chunks (prevents loss of context)
    length_function=len,   # How to measure the length (standard Python len)
)
  main_placeholder.text("Text Splitter...Started....✅✅✅")
  docs = recursive_text_splitter.split_documents(data)
  #create embeddings and save it to FAISS index
  with st.spinner("Generating embeddings... this may take a minute."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedded Vector started building....✅✅✅")
  st.success("Embeddings created successfully!")
  vectorindex = FAISS.from_documents(docs, embeddings)
  time.sleep(2)
  # Save the index to a folder named "my_faiss_index"
  vectorindex.save_local("my_faiss_index")


with st.form(key='question_form'):
    query = st.text_input("Question : ")
    submit_button = st.form_submit_button(label='Ask')

if submit_button and query:
  print("This is test-2")
  if os.path.exists(file_path) and os.path.isdir(file_path):
    # Load the existing index
    new_db = FAISS.load_local(
        file_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, 
        retriever=new_db.as_retriever(), 
        verbose=True
    )
    result = chain.invoke({"question": query}, return_only_outputs=True)
    st.header("Answer")
    st.subheader(result["answer"])

    #Display sources, if available
    sources = result.get("sources", "")
    if sources:
      st.subheader("Sources:")
      sources_list = sources.split("\n")
      for source in sources_list:
        st.write(source)