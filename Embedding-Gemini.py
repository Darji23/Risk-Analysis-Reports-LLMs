import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.document_loaders.base import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(current_dir, "reports")
analysis_dir = os.path.join(current_dir, "analyses")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_gemini")


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(reports_dir):
        raise FileNotFoundError(
            f"The directory {reports_dir} does not exist. Please check the path."
        )
    
    if not os.path.exists(analysis_dir):
        raise FileNotFoundError(
            f"The directory {analysis_dir} does not exist. Please check the path."
        )

    report_files = sorted([f for f in os.listdir(reports_dir) if f.endswith(".pdf")])
    analysis_files = sorted([f for f in os.listdir(analysis_dir) if f.endswith(".pdf")])

    if len(report_files) != len(analysis_files):
        raise ValueError("The number of audit reports does not match the number of risk analyses.")
    else:
        print("------------Files Found Successfully---------")


    print("-------Starting Extraction-----------")
    documents = []

    # Assuming you want to generate a unique ID for each report-analysis pair
    for index, (report_file, analysis_file) in enumerate(zip(report_files, analysis_files)):
        report_path = os.path.join(reports_dir, report_file)
        analysis_path = os.path.join(analysis_dir, analysis_file)

        report_text = extract_text_from_pdf(report_path)
        analysis_text = extract_text_from_pdf(analysis_path)

        # Generate a unique audit_id for each pair of audit report and risk analysis report
        audit_id = f"audit_{index + 1}"

        # Create document objects with the audit_id in the metadata
        report_doc = Document(
            page_content=report_text, 
            metadata={"source": report_file, "type": "audit_report", "audit_id": audit_id}
        )
        analysis_doc = Document(
            page_content=analysis_text, 
            metadata={"source": analysis_file, "type": "risk_analysis", "audit_id": audit_id}
        )

        documents.append(report_doc)
        documents.append(analysis_doc)
    
    print("---------Extraction Completed-------------")

    print("------------Starting to Split--------------")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print("------------Splitting Completed-------------")

    print("----Document Chunks Information-----")
    print(f"Number of document chunks:{len(docs)}")

    print("-----Creating Embeddings-----")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("-----Finished Creating Embeddings-----")

    print("------Creating and persisting vector store----")
    db=Chroma.from_documents(docs,embeddings,persist_directory=persistent_directory)
    print("-----Finished Creating and Persisting Vector Store-----")

else:
    print("Vector store already exists. No need to initilaize")


