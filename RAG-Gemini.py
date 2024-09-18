import os 
from dotenv import load_dotenv
import pdfplumber
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datasets import Dataset 
from ragas.metrics import faithfulness
from ragas import evaluate

load_dotenv()

print("-----Initializing----")
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_gemini")
output_dir = os.path.join(current_dir,"test_case/Output/Gemini")
print("------Created Paths Successfully-------")

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

pdf_path = os.path.join(current_dir, "test_case/TCS-Q4.pdf")
user_audit_report = extract_text_from_pdf(pdf_path)
if not user_audit_report:
    print("No text extracted from Testcase File")
else:
    print(f"Testcase File text extracted successfully!!")

pdf_path = os.path.join(current_dir, "test_case/TCS-Q4-Ground-Report.pdf")
ground_truth_audit_report = extract_text_from_pdf(pdf_path)
if not ground_truth_audit_report:
    print("No text extracted from Ground Truth File")
else:
    print(f"GroundTruth File text extracted successfully!!")

print("------STARTING TO RETRIEVE--------")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})  
print("------DOCUMENTS RETRIEVED------")

print("------FINDING RELEVANT DOCUMENTS-------")
relevant_docs = retriever.invoke(user_audit_report)
print("--------RELEVANT DOCUMENTS FOUND SUCESSFULLY---------")

if not relevant_docs:
    print("No relevant documents found.")
else:
    matched_risk_analyses = []
    matched_audit_reports = []
    all_docs = db.get()
    for doc in relevant_docs:
        audit_id = doc.metadata.get("audit_id")
        if doc.metadata.get("type") == "audit_report":
            print(f"Document_type: Audit Report")
            matched_audit_reports.append(doc.page_content)
            print("Relevant Audit Report Appended Succesfully!!")
        
        for doc_id, document, metadata in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
            if metadata.get('audit_id') == audit_id and metadata.get('type') == 'risk_analysis':
                print(f"Document_type: Risk Analysis Report")
                matched_risk_analyses.append(document)
                print("Relevant Risk Analysis Report Appended Succesfully!!")


combined_audit_reports = "\n\n".join(matched_audit_reports)
combined_risk_analyses = "\n\n".join(matched_risk_analyses)

combined_input = (
    "The user have given quarterly financial results of a company. Learn the data provided in the file"
    + user_audit_report
    +"First tell me the name of the company specified in the user given quarterly financial report."
    + "\n\n The relevant risk analysis reports are :\n"
    + combined_risk_analyses
    + "\n\nAssess the quarterly financial results given by the user. Create a risk analysis report of the user's quarterly financial results using the relevant risk analysis documents by using its terminologies,format,semantics,writing style etc..."
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

messages = [
    SystemMessage(content="Create a risk analysis report of the quarterly financial results provided by the user and if your find any difficulty, do tell me."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("------Generated Response-------\n\n")
print("THE RISK ANALYSIS REPORT GENERATED:-")
generated_risk_analysis = result.content
print(generated_risk_analysis)  
print("\n\n\n-------REPORT GENERATED SUCESSFULLY---------")

output_file_path = os.path.join(output_dir, "Generated_Response_Q4.txt")

try:
    with open(output_file_path, "w",encoding="utf-8") as file:
        file.write(generated_risk_analysis)
    print(f"\n\nGenerated risk analysis report successfully saved to {output_file_path}")
except Exception as e:
    print(f"Error saving generated report: {str(e)}")


# Evaluation metrics

def calculate_faithfulness(generated_text, source_text):
    key_phrases = re.findall(r'\b\w+\b', source_text.lower())
    generated_phrases = re.findall(r'\b\w+\b', generated_text.lower())
    
    faithful_phrases = set(key_phrases) & set(generated_phrases)
    if not key_phrases:
        print("Warning: No key phrases found in source text.")
        return 0
    faithfulness_score = len(faithful_phrases) / len(set(key_phrases))
    
    return faithfulness_score

def calculate_context_precision_recall(generated_text, context):
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    generated_words = set(re.findall(r'\b\w+\b', generated_text.lower()))
    
    common_words = context_words & generated_words
    
    precision = len(common_words) / len(generated_words) if generated_words else 0
    recall = len(common_words) / len(context_words) if context_words else 0
    
    return precision, recall

def calculate_relevancy(text, reference):
    vectorizer = TfidfVectorizer().fit_transform([text, reference])
    cosine_sim = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
    return cosine_sim

def calculate_answer_correctness(generated_text, ground_truth):
    vectorizer = TfidfVectorizer().fit_transform([generated_text, ground_truth])
    similarity = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
    return similarity


# Calculate and print evaluation metrics
print("\n----- Evaluation Metrics -----")

try:
    faithfulness = calculate_faithfulness(generated_risk_analysis, combined_risk_analyses)
    print(f"Faithfulness: {faithfulness:.4f}")
except Exception as e:
    print(f"Error calculating faithfulness: {str(e)}")

try:
    context_precision, context_recall = calculate_context_precision_recall(generated_risk_analysis, combined_risk_analyses)
    print(f"Context Precision: {context_precision:.4f}")
    print(f"Context Recall: {context_recall:.4f}")
except Exception as e:
    print(f"Error calculating context precision/recall: {str(e)}")

try:
    answer_relevancy = calculate_relevancy(generated_risk_analysis, user_audit_report)
    print(f"Answer Relevancy: {answer_relevancy:.4f}")
except Exception as e:
    print(f"Error calculating answer relevancy: {str(e)}")

try:
    context_relevancy = calculate_relevancy(generated_risk_analysis, combined_risk_analyses)
    print(f"Context Relevancy: {context_relevancy:.4f}")
except Exception as e:
    print(f"Error calculating context relevancy: {str(e)}")

try:
    answer_correctness = calculate_answer_correctness(generated_risk_analysis, ground_truth_audit_report)
    print(f"Answer Correctness: {answer_correctness:.4f}")
except Exception as e:
    print(f"Error calculating answer correctness: {str(e)}")
