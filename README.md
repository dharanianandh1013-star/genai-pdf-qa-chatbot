## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
In many cases, users need specific information from large documents without manually searching through them. A question-answering chatbot can address this problem by:

    Parsing and indexing the content of a PDF document.
    Allowing users to ask questions in natural language.
    Providing concise and accurate answers based on the content of the document.

The implementation will evaluate the chatbot’s ability to handle diverse queries and deliver accurate responses.

### DESIGN STEPS

#### STEP 1: 
Load and Parse PDF
Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: 
Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: 
Initialize the LangChain QA Pipeline
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

#### STEP 4: 
Handle User Queries
Process user queries, retrieve relevant document sections, and generate responses.

#### STEP 5: 
Evaluate Effectiveness
Test the chatbot with a variety of queries to assess accuracy and reliability.


### PROGRAM:
```
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

pdf = canvas.Canvas("summary of the project.pdf", pagesize=A4)

text = pdf.beginText(50, 800)
text.setFont("Helvetica", 12)

content = [
    "PROJECT: LIBRARY MANAGEMENT SYSTEM",
    "",
    "OBJECTIVE:",
    "The main objective is to develop a Library Management System",
    "that simplifies book management and library operations.",
    "",
    "COMPONENTS USED:",
    "1. User Registration",
    "2. Book Management",
    "3. Book Search",
    "4. Book Issue and Return",
    "5. Database Management",
    "",
    "WORKING:",
    "Users can register and log in to the system.",
    "They can search for books using book names or authors.",
    "The librarian can add, update and delete book information.",
    "Students can borrow and return books.",
    "The database stores user details and book records.",
    "",
    "CORE INNOVATION:",
    "The system automates library operations and reduces manual work.",
    "It provides quick book searching and efficient record management.",
    "",
    "ADVANTAGES:",
    "1. Saves time.",
    "2. Reduces paperwork.",
    "3. Improves book management.",
    "4. Provides easy access to book information."
]

for line in content:
    text.textLine(line)

pdf.drawText(text)
pdf.save()

print("Sample PDF created successfully!")
```

```
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found. Check your .env file.")

loader = PyPDFLoader("summary of the project.pdf")
pages = loader.load()

print("PDF loaded successfully")
print("Total pages:", len(pages))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

print("Total chunks:", len(chunks))

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="docs/chroma",
    collection_name="project_pdf"
)

print("Vector database created successfully")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True
)

print("\nPDF Question Answering Chatbot is ready!")

while True:
    question = input("\nAsk a question (type exit to stop): ")

    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    result = qa.invoke({"question": question})

    print("\nAnswer:", result["answer"])
```

### OUTPUT:
<img width="1298" height="830" alt="image" src="https://github.com/user-attachments/assets/673f8d79-0f34-4a3e-8357-58c4ce7a14e6" />

### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
