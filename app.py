import streamlit as st
import pickle
from PyPDF2 import PdfReader
import docx
import os
from io import BytesIO
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_extras.add_vertical_space import add_vertical_space


#Sidebar Content
with st.sidebar:
    st.title('Query Your Document')
    st.markdown('''
    This app is an LLM-Powered ChatBot built using:
    - [StreamLit](https://docs.streamlit.io/)
    - [LangChain](https://python.langchain.com)
    - [OpenAI](https://platform.openai.com/docs/models)


    ''')
    add_vertical_space(5)
    st.write("Made by Ishan Kundekar")

# Function to read content from an uploaded .docx file
def read_word_from_streamlit(doc):
    # Convert the uploaded file (in memory) to a BytesIO object
    docx_file = BytesIO(doc.read())
    # Open the docx file using python-docx
    document = docx.Document(docx_file)
    # Extract text from all paragraphs in the document
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    
    return text


def main():
    st.header("Talk to your Document")
    load_dotenv()

    #Upload your Document
    file= st.file_uploader("Upload your document", type=['pdf', 'docx']) #Limit of 200mb
    
    if file is not None:
        
        if file.type=='application/pdf':  
            pdf_reader=PdfReader(file)  
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        else: 
            #Use the function on a .docx fil
            document_text = read_word_from_streamlit(file)
        
        
        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1000,#These are the number of input tokens for embedding
            chunk_overlap=200,#These are the tokens that will be overlapped
            length_function=len,
            )
        if file.type=='application/pdf':
            chunks = text_splitter.split_text(text=text)
        else:
            chunks = text_splitter.split_text(text=document_text)
        
        embeddings=OpenAIEmbeddings()
        
         
        #Creating Embeddings 
        
        store_name=file.name[:-4]
        
        #if os.path.exists(f"{store_name}.pkl"):
         #   with open(f"{store_name}.pkl","rb") as f:
          #      VectorStore=pickle.load(f)
           # st.write('Embeddings loaded from the disk')
        if os.path.exists(f"{store_name}_faiss"):
            
            VectorStore = FAISS.load_local(f"{store_name}_faiss",embeddings, allow_dangerous_deserialization=True)
            st.write("Embeddings loaded from disk successfully.")
        else:
            
            #with open(f"{store_name}.pkl","wb") as f:
                #pickle.dump(VectorStore,f)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(f"{store_name}_faiss")
            st.write("Embeddings Computation Successful")

        #Accept query from the user
        query = st.text_input("Ask questions about your file:")
        st.write(query)

        if query:
            docs= VectorStore.similarity_search(query=query , k=3)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            chain= load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response= chain.run(input_documents=docs, question= query)
                print(cb)
            st.write(response)

    

if __name__ == '__main__':
    main()