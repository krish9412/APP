import streamlit as st
import os
import tempfile
import json
import io
import pdfplumber
import uuid
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# LangChain imports for document retrieval and embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Import FAISS instead of Chroma to avoid protobuf issues
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Configure the Streamlit page layout
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Initialize session state variables for managing app state
if 'course_data' not in st.session_state:
    st.session_state.course_data = None
if 'course_ready' not in st.session_state:
    st.session_state.course_ready = False
if 'generating_course' not in st.session_state:
    st.session_state.generating_course = False
if 'answered_questions' not in st.session_state:
    st.session_state.answered_questions = set()
if 'total_quiz_questions' not in st.session_state:
    st.session_state.total_quiz_questions = 0
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'queries_list' not in st.session_state:
    st.session_state.queries_list = []
if 'unique_session' not in st.session_state:
    st.session_state.unique_session = str(uuid.uuid4())
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
if 'uploaded_doc_names' not in st.session_state:
    st.session_state.uploaded_doc_names = []
if 'doc_vector_db' not in st.session_state:
    st.session_state.doc_vector_db = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Sidebar setup
st.sidebar.title("🎓 Professional Learning System")

# Reset button to clear session state
if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.unique_session = str(uuid.uuid4())
    st.session_state.processed_docs = []
    st.session_state.uploaded_docs = []
    st.session_state.uploaded_doc_names = []
    st.session_state.doc_vector_db = None
    st.session_state.embedding_model = None
    st.session_state.queries_list = []
    st.rerun()

# Input for OpenAI API key, with support for secrets.toml
api_key = st.secrets["openai"]["api_key"] if "openai" in st.secrets else st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# File uploader for PDFs
pdf_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            extracted_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                extracted_text += text + "\n"
        return extracted_text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return ""

# Process uploaded PDFs and store in session state
if pdf_files and api_key:
    current_doc_names = [file.name for file in pdf_files]
    
    # Check if new documents were uploaded
    if current_doc_names != st.session_state.uploaded_doc_names:
        st.session_state.processed_docs = []
        st.session_state.uploaded_docs = []
        st.session_state.uploaded_doc_names = current_doc_names
        st.session_state.doc_vector_db = None
        doc_list = []
        
        with st.spinner("Processing your PDF documents..."):
            for pdf_file in pdf_files:
                text = extract_text_from_pdf(pdf_file)
                if text:
                    st.session_state.processed_docs.append({
                        "name": pdf_file.name,
                        "content": text
                    })
                    st.session_state.uploaded_docs.append(pdf_file)
                    doc_list.append(Document(page_content=text, metadata={"name": pdf_file.name}))
                    
        if st.session_state.processed_docs:
            st.sidebar.success(f"✅ {len(st.session_state.processed_docs)} PDFs processed successfully!")
            
            try:
                # Create chunks for better retrieval
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                doc_chunks = splitter.split_documents(doc_list)
                
                # Initialize embedding model only once
                if st.session_state.embedding_model is None:
                    st.session_state.embedding_model = OpenAIEmbeddings(
                        api_key=api_key,
                        model="text-embedding-ada-002"
                    )
                
                # Create vector database using FAISS instead of Chroma
                st.session_state.doc_vector_db = FAISS.from_documents(
                    documents=doc_chunks,
                    embedding=st.session_state.embedding_model
                )
                
                st.sidebar.success("✅ Vector database initialized successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to initialize vector database: {str(e)}")
                st.session_state.doc_vector_db = None
else:
    if not api_key:
        st.info("📥 Please provide your OpenAI API key to start.")
    elif not pdf_files:
        st.info("📥 Please upload PDFs to start.")

# Model and role selection in sidebar
model_choices = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_choices, index=0)

role_choices = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
user_role = st.sidebar.selectbox("Select Your Role", role_choices)

focus_areas = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
selected_focus = st.sidebar.multiselect("Select Learning Focus", focus_areas)

# Display uploaded files in sidebar
if st.session_state.uploaded_doc_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for idx, doc_name in enumerate(st.session_state.uploaded_doc_names):
        st.sidebar.text(f"{idx+1}. {doc_name}")

# Function to generate answers using retrieved documents
def answer_with_retrieval(query):
    try:
        if not api_key:
            return "Please provide an OpenAI API key to proceed."
        
        if not st.session_state.doc_vector_db:
            return "Please upload and process documents first to enable answering questions."
            
        # Retrieve relevant document chunks with similarity scores - adapted for FAISS
        search_results = st.session_state.doc_vector_db.similarity_search_with_score(query, k=3)
        
        # Check if any documents were retrieved
        if not search_results:
            return "No relevant information found in the uploaded documents for this query."
        
        # Extract documents and their scores
        relevant_docs = [result[0] for result in search_results]
        scores = [result[1] for result in search_results]
        
        # Build the document context
        doc_context = ""
        for doc in relevant_docs:
            doc_context += f"\nSource: {doc.metadata.get('name', 'Unknown')}\nContent: {doc.page_content}\n"
        
        # Construct the prompt
        full_prompt = (
            "You are an AI assistant for a professional learning platform. Provide a detailed and accurate answer to the following query "
            "using ONLY the document excerpts provided below. Be thorough and reference the documents where applicable.\n\n"
            f"Query: {query}\n\n"
            f"Document Excerpts:\n{doc_context}\n\n"
            "Answer the query comprehensively using only the provided document excerpts. If the information is insufficient, state so politely."
        )
        
        # Initialize and call the LLM
        llm = ChatOpenAI(api_key=api_key, model=selected_model, temperature=0.5)
        response = llm.invoke(full_prompt)
        answer_text = response.content
        
        # Add references to the answer
        if relevant_docs:
            answer_text += "\n\n**Sources Referenced:**\n"
            for doc in relevant_docs:
                doc_name = doc.metadata.get("name", "Unknown")
                answer_text += f"- {doc_name}\n"
        
        return answer_text
    except Exception as e:
        return f"Failed to generate answer: {str(e)}"

# Sidebar section for employer queries
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_question = st.sidebar.text_area("Add a new question (related to uploaded documents):", height=100)
if st.sidebar.button("Submit Question"):
    if new_question:
        with st.spinner("Generating response..."):
            try:
                response = answer_with_retrieval(new_question)
            except Exception as e:
                response = f"Failed to generate response: {str(e)}"
        
        st.session_state.queries_list.append({
            "question": new_question,
            "response": response,
            "answered": True
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Function to validate quiz answers
def validate_answer(q_id, user_response, correct_response):
    user_response_str = str(user_response).strip().lower() if user_response is not None else ""
    correct_response_str = str(correct_response).strip().lower() if correct_response is not None else ""
    
    if user_response_str == correct_response_str:
        st.success("🎉 Correct answer! Great job!")
        st.session_state.answered_questions.add(q_id)
        return True
    else:
        st.error(f"Incorrect. The correct answer is: {correct_response}")
        return False

# Function to generate a progress report PDF
def create_progress_report():
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    page_width, page_height = letter

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(page_width / 2, page_height - 50, "Professional Learning Platform")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(page_width / 2, page_height - 70, "Training Progress Report")
    pdf.line(50, page_height - 80, page_width - 50, page_height - 80)

    pdf.setFont("Helvetica", 12)
    y_pos = page_height - 110

    pdf.drawString(50, y_pos, f"User Role: {user_role}")
    y_pos -= 20
    pdf.drawString(50, y_pos, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    y_pos -= 20

    if st.session_state.course_data:
        course_name = st.session_state.course_data.get('course_title', 'Not Available')
        pdf.drawString(50, y_pos, f"Course: {course_name}")
        y_pos -= 20
    if selected_focus:
        pdf.drawString(50, y_pos, f"Learning Focus: {', '.join(selected_focus)}")
        y_pos -= 20

    y_pos -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y_pos, "Progress Overview:")
    pdf.setFont("Helvetica", 12)
    y_pos -= 20
    completed_count = len(st.session_state.answered_questions)
    total_count = st.session_state.total_quiz_questions
    progress_percent = (completed_count / total_count * 100) if total_count > 0 else 0
    pdf.drawString(50, y_pos, f"Questions Answered: {completed_count}/{total_count}")
    y_pos -= 20
    pdf.drawString(50, y_pos, f"Progress: {progress_percent:.1f}%")
    y_pos -= 20

    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawCentredString(page_width / 2, 30, f"Generated by Professional Learning Platform on {datetime.now().strftime('%Y-%m-%d')}")
    
    pdf.showPage()
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Function to initiate course generation
def start_course_creation():
    st.session_state.generating_course = True
    st.session_state.course_ready = False
    st.rerun()

# Function to generate the course content
def generate_course_content():
    try:
        combined_content = ""
        for idx, doc in enumerate(st.session_state.processed_docs):
            doc_text = f"\n--- Document {idx+1}: {doc['name']} ---\n"
            doc_text += doc['content'][:3000]
            combined_content += doc_text + "\n\n"
        
        user_context = f"Role: {user_role}, Focus Areas: {', '.join(selected_focus)}"
        
        doc_summary_query = "Summarize these documents comprehensively, focusing on key concepts."
        doc_summary = answer_with_retrieval(doc_summary_query)
        
        course_prompt = f"""
        Create a detailed professional learning course based on the provided documents.
        User Context: {user_context}
        Document Summary: {doc_summary}
        
        Document Content: {combined_content[:5000]}
        
        Develop a structured course with:
        1. A compelling course title
        2. A course description (300+ words)
        3. 5-8 logical modules
        4. 4-6 learning objectives per module
        5. Detailed module content (500+ words each)
        6. A quiz per module (3-5 questions)
        
        For each quiz question:
        - Provide multiple choice options (A, B, C, D)
        - Specify the correct answer letter
        
        Return in JSON format with course_title, course_description, and modules.
        """
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": course_prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        course_json = response.choices[0].message.content
        try:
            st.session_state.course_data = json.loads(course_json)
            st.session_state.course_ready = True
            total_questions = 0
            for module in st.session_state.course_data.get("modules", []):
                quiz = module.get("quiz", {})
                total_questions += len(quiz.get("questions", []))
            st.session_state.total_quiz_questions = total_questions
        except json.JSONDecodeError as e:
            st.error(f"Error parsing course JSON: {e}")
            st.text(course_json)
        
    except Exception as e:
        st.error(f"Course generation failed: {e}")
    
    st.session_state.generating_course = False

# Main tabs for navigation
content_tab, queries_tab, docs_tab = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

if st.session_state.generating_course:
    with st.spinner("Creating your personalized course..."):
        st.session_state.answered_questions = set()
        generate_course_content()
        st.success("✅ Course Generated Successfully!")
        st.rerun()

with content_tab:
    if st.session_state.course_ready and st.session_state.course_data:
        course = st.session_state.course_data
        
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.markdown(f"*Designed for {user_role}s focusing on {', '.join(selected_focus)}*")
        st.write(course.get('course_description', 'A course to boost your professional skills.'))
        
        completed = len(st.session_state.answered_questions)
        total = st.session_state.total_quiz_questions
        progress = (completed / total * 100) if total > 0 else 0
        
        st.progress(progress / 100)
        st.write(f"**Progress:** {completed}/{total} questions completed ({progress:.1f}%)")
        st.download_button("📥 Download Progress Report", create_progress_report(), "progress_report.pdf")
        
        st.markdown("---")
        
        modules = course.get("modules", [])
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                for obj in objectives:
                    st.markdown(f"- {obj}")
                
                st.markdown("### 📖 Module Content:")
                content = module.get('content', 'No content available.')
                st.write(content)
                
                st.markdown("### 📝 Module Quiz:")
                quiz = module.get("quiz", {})
                questions = quiz.get("questions", [])
                
                for q_idx, q in enumerate(questions, 1):
                    q_id = f"module_{i}_question_{q_idx}"
                    q_text = q.get('question', f'Question {q_idx}')
                    
                    with st.container():
                        st.markdown(f"**Question {q_idx}:** {q_text}")
                        options = q.get('options', [])
                        
                        if options:
                            option_key = f"quiz_{i}_{q_idx}"
                            user_answer = st.radio(
                                "Select your answer:", 
                                options, 
                                key=option_key,
                                index=None
                            )
                            
                            submit_key = f"submit_{i}_{q_idx}"
                            if q_id in st.session_state.answered_questions:
                                st.success("✓ Question completed")
                            else:
                                if st.button(f"Check Answer", key=submit_key):
                                    correct_answer = q.get('correct_answer', '')
                                    user_letter = user_answer[0] if user_answer and len(user_answer) > 0 else ""
                                    validate_answer(q_id, user_letter, correct_answer)
                        
                        st.markdown("---")

    else:
        st.title("Welcome to Professional Learning Platform")
        st.markdown("""
        ## Elevate your professional skills with AI-driven learning
        
        Upload your PDF documents, and I'll craft a tailored course for you!
        
        ### Steps to Begin:
        1. Ensure your OpenAI API key is set up
        2. Select your role and focus areas
        3. Upload PDF documents
        4. Click "Generate Course" to start learning
        """)
        
        if st.session_state.processed_docs and api_key and not st.session_state.generating_course:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate My Course", use_container_width=True):
                    start_course_creation()
        elif st.session_state.generating_course:
            st.info("Generating your course... Please wait.")

with queries_tab:
    st.title("💬 Employer Queries")
    st.markdown("""
    Employers can ask questions here to get AI-generated insights based on the uploaded documents.
    Submit your query in the sidebar, and the AI will respond with answers derived from the document content.
    """)
    
    # Check if the document vector database is ready
    if not st.session_state.doc_vector_db:
        st.warning("⚠️ No documents processed yet. Please upload and process PDF documents to enable answering questions.")
    
    if not st.session_state.queries_list:
        st.info("No queries submitted yet. Add a question in the sidebar to begin.")
    else:
        for idx, query in enumerate(st.session_state.queries_list):
            with st.expander(f"Question {idx+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {idx+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                if query.get('answered'):
                    st.write(f"**Answer:** {query['response']}")
                else:
                    st.info("Processing answer...")
                    # This part should never execute since we set 'answered' to True when adding to queries_list
                    with st.spinner("Generating answer..."):
                        response = answer_with_retrieval(query['question'])
                        st.session_state.queries_list[idx]['response'] = response
                        st.session_state.queries_list[idx]['answered'] = True
                        st.rerun()

with docs_tab:
    st.title("📑 Document Sources")
    
    if not st.session_state.processed_docs:
        st.info("No documents uploaded yet. Upload PDFs in the sidebar to view them here.")
    else:
        st.write(f"**{len(st.session_state.processed_docs)} documents uploaded:**")
        
        for idx, doc in enumerate(st.session_state.processed_docs):
            with st.expander(f"Document {idx+1}: {doc['name']}"):
                preview = doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content']
                st.markdown("### Document Preview:")
                st.text_area("Content Preview:", value=preview, height=300, disabled=True)
                
                if st.button(f"Generate Summary for {doc['name']}", key=f"summary_{idx}"):
                    with st.spinner("Generating summary..."):
                        summary_query = f"Summarize the document '{doc['name']}'"
                        summary = answer_with_retrieval(summary_query)
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
