import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangchainOpenAI

# Page Configuration
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Initialize session state variables
if 'course_content' not in st.session_state:
    st.session_state.course_content = None
if 'course_generated' not in st.session_state:
    st.session_state.course_generated = False
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'completed_questions' not in st.session_state:
    st.session_state.completed_questions = set()
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []
if 'employer_queries' not in st.session_state:
    st.session_state.employer_queries = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""


# --- Helper Functions ---
def extract_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""


def generate_progress_report():
    """Generate a PDF progress report."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Training Progress Report")
    c.drawString(100, 730, f"User Role: {st.session_state.get('role', 'Not specified')}")
    c.drawString(100, 710, f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    
    # Avoid division by zero
    progress_percentage = (completed / total * 100) if total > 0 else 0
    c.drawString(100, 690, f"Progress: {completed}/{total} questions completed ({progress_percentage:.1f}%)")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


def check_answer(question_id, user_answer, correct_answer):
    """Check if the user's answer is correct."""
    if user_answer == correct_answer:
        st.success("🎉 Correct! Well done!")
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False


# --- RAG Functions ---
def create_vectorstore():
    """Creates and stores the vectorstore in the session state."""
    try:
        if not st.session_state.extracted_texts:
            st.warning("No text has been extracted from documents yet.")
            return False
            
        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API key.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        embeddings_model = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)

        docs = []
        for doc in st.session_state.extracted_texts:
            chunks = text_splitter.split_text(doc["text"])
            docs.extend(chunks)

        if not docs:
            st.warning("No text chunks were created from the documents.")
            return False

        st.session_state.vectorstore = Chroma.from_texts(docs, embeddings_model)
        st.success("✅ Vectorstore created successfully!")
        return True
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return False


def generate_rag_answer(question):
    """Generates an answer using Retrieval Augmented Generation."""
    try:
        if not st.session_state.vectorstore:
            return "Please upload and process documents to generate answers."
            
        if not st.session_state.openai_api_key:
            return "Please enter your OpenAI API key."

        llm = LangchainOpenAI(
            openai_api_key=st.session_state.openai_api_key, 
            model_name=st.session_state.get('selected_model', 'gpt-4o-mini')
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        result = qa_chain({"query": question})
        return result["result"]
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return f"An error occurred: {str(e)}"


def perform_course_generation():
    """Generates the course content using the LLM and RAG."""
    try:
        if not st.session_state.vectorstore:
            raise ValueError("Vectorstore not initialized. Please upload and process documents first.")
            
        if not st.session_state.openai_api_key:
            raise ValueError("Please enter your OpenAI API key.")

        # Prepare context for the LLM
        combined_docs = ""
        for i, doc in enumerate(st.session_state.extracted_texts):
            doc_summary = f"\n--- DOCUMENT {i + 1}: {doc['filename']} ---\n"
            doc_summary += doc['text'][:3000]  # Limit to 3000 chars per document
            combined_docs += doc_summary + "\n\n"

        professional_context = (f"Role: {st.session_state.get('role', 'Professional')}, "
                              f"Focus: {', '.join(st.session_state.get('learning_focus', ['Professional Development']))}")

        # Get document summary using RAG
        summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
        document_summary = generate_rag_answer(summary_query)

        prompt = f"""
        Design a comprehensive professional learning course based on the multiple documents provided.
        Context: {professional_context}
        Document Summary: {document_summary}

        Document Contents: {combined_docs[:5000]}

        Create an engaging, thorough and well-structured course by:
        1. Analyzing all provided documents and identifying common themes, complementary concepts, and unique insights from each source
        2. Creating an inspiring course title that reflects the integrated knowledge from all documents
        3. Writing a detailed course description (at least 300 words) that explains how the course synthesizes information from multiple sources
        4. Developing 5-8 comprehensive modules that build upon each other in a logical sequence
        5. Providing 4-6 clear learning objectives for each module with specific examples and practical applications
        6. Creating detailed, well-explained content for each module (at least 500 words per module) including:
           - Real-world examples and case studies
           - Practical applications of concepts
           - Visual explanations where appropriate
           - Step-by-step guides for complex procedures
           - Comparative analysis when sources present different perspectives
        7. Including a quiz with 3-5 thought-provoking questions per module for better understanding

        Return the response in the following JSON format:
        {{
            "course_title": "Your Course Title",
            "course_description": "Detailed description of the course",
            "modules": [
                {{
                    "title": "Module 1 Title",
                    "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"],
                    "content": "Module content text with detailed explanations, examples, and practical applications",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Question text?",
                                "options": ["Option A", "Option B", "Option C", "Option D"],
                                "correct_answer": "Option A"
                            }}
                        ]
                    }}
                }}
            ]
        }}

        Make the content exceptionally practical, actionable, and tailored to the professional context.
        Provide detailed explanations, real-world examples, and practical applications in each module content.
        Where document sources provide different perspectives or approaches to the same topic, compare and contrast them.
        """

        client = OpenAI(api_key=st.session_state.openai_api_key)
        response = client.chat.completions.create(
            model=st.session_state.get('selected_model', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        response_content = response.choices[0].message.content

        try:
            st.session_state.course_content = json.loads(response_content)
            st.session_state.course_generated = True

            total_questions = 0
            for module in st.session_state.course_content.get("modules", []):
                quiz = module.get("quiz", {})
                total_questions += len(quiz.get("questions", []))
            st.session_state.total_questions = total_questions

        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {e}")
            st.text(response_content)

    except Exception as e:
        st.error(f"Error generating course: {e}")

    st.session_state.is_generating = False


def generate_course():
    """Initiates course generation."""
    if not st.session_state.vectorstore:
        st.error("Please upload and process documents before generating a course.")
        return
        
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key.")
        return
        
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()


# --- Streamlit UI ---

# Sidebar Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# 🔐 OpenAI API Key Input
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")
if openai_api_key:
    st.session_state.openai_api_key = openai_api_key

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'],
                                          accept_multiple_files=True)

# Process uploaded files
if uploaded_files and st.session_state.openai_api_key:
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames

        with st.spinner("Processing PDF files..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)

        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")

            # Create the vectorstore after processing files
            create_vectorstore()
elif not st.session_state.openai_api_key:
    st.info("📥 Please enter your OpenAI API key to begin.")
elif not uploaded_files:
    st.info("📥 Please upload PDF files to begin.")

# 🎯 GPT Model and Role selection
model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0)
if selected_model:
    st.session_state.selected_model = selected_model

role_options = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources",
                "Other", "Fresher"]
role = st.sidebar.selectbox("Select Your Role", role_options)
if role:
    st.session_state.role = role

learning_focus_options = ["Leadership", "Technical Skills", "Communication", "Project Management",
                          "Innovation", "Team Building", "Finance"]
learning_focus = st.sidebar.multiselect("Select Learning Focus", learning_focus_options)
if learning_focus:
    st.session_state.learning_focus = learning_focus

# Display uploaded files in sidebar
if st.session_state.uploaded_file_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for i, filename in enumerate(st.session_state.uploaded_file_names):
        st.sidebar.text(f"{i + 1}. {filename}")

# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        answer = ""
        if st.session_state.vectorstore and st.session_state.openai_api_key:
            with st.spinner("Generating answer..."):
                answer = generate_rag_answer(new_query)
        elif not st.session_state.vectorstore:
            answer = "Please upload and process documents first to enable question answering."
        elif not st.session_state.openai_api_key:
            answer = "Please enter your OpenAI API key."

        st.session_state.employer_queries.append({
            "question": new_query,
            "answer": answer,
            "answered": bool(answer)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Generate Course Button (moved up for better visibility)
if st.sidebar.button("🚀 Generate Course"):
    if st.session_state.extracted_texts and st.session_state.openai_api_key:
        generate_course()
    elif not st.session_state.extracted_texts:
        st.sidebar.error("Please upload PDF files first.")
    elif not st.session_state.openai_api_key:
        st.sidebar.error("Please enter your OpenAI API key.")

# Main contents area with tabs
tab1, tab2, tab3 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

# Check if we're in the middle of generating a course
if st.session_state.is_generating:
    with st.spinner("Generating your personalized course from multiple documents..."):
        st.session_state.completed_questions = set()  # Reset progress
        perform_course_generation()
        st.success("✅ Your Comprehensive Course is Ready!")
        st.rerun()

with tab1:
    # Display Course Content
    if not st.session_state.course_generated:
        st.info("📚 Upload PDFs and click 'Generate Course' to create a personalized learning experience.")
        
        # Display a demo button if no course is generated yet
        if st.button("🔍 See Example Course"):
            st.markdown("""
            ### Example Course: Strategic Leadership in Digital Transformation
            
            This course would integrate concepts from your uploaded materials, covering:
            
            - **Module 1**: Digital Leadership Foundations
            - **Module 2**: Strategic Change Management
            - **Module 3**: Building High-Performance Teams
            - **Module 4**: Innovation and Digital Solutions
            - **Module 5**: Measuring Success and ROI
            
            Upload your PDFs and generate a custom course tailored to your specific needs!
            """)
    
    elif st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content

        # Course Header
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.markdown(f"*Specially designed for {st.session_state.get('role', 'Professionals')} focusing on {', '.join(st.session_state.get('learning_focus', ['Professional Development']))}*")
        st.write(course.get('course_description', 'A structured course to enhance your skills.'))

        # Progress Tracking
        completed = len(st.session_state.completed_questions)
        total = st.session_state.total_questions
        progress_percentage = (completed / total * 100) if total > 0 else 0

        st.progress(progress_percentage / 100)
        st.write(f"**Progress:** {completed}/{total} questions completed ({progress_percentage:.1f}%)")
        st.download_button("📥 Download Progress Report", generate_progress_report(),
                           "progress_report.pdf")

        st.markdown("---")
        st.subheader("📋 Course Overview")

        modules = course.get("modules", [])
        if modules:
            modules_list = [module.get('title', f'Module {i + 1}') for i, module in enumerate(modules)]
            for i, module_title in enumerate(modules_list, 1):
                st.write(f"**Module {i}:** {module_title}")
        else:
            st.warning("No modules were found in the course content.")

        st.markdown("---")

        # Detailed Module Contents
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives specified.")

                st.markdown("### 📖 Module Content:")
                module_content = module.get('content', 'No content available for this module.')

                paragraphs = module_content.split('\n\n')
                for para in paragraphs:
                    if para.strip().startswith('#'):
                        st.markdown(para)
                    elif para.strip().startswith('*') and para.strip().endswith('*'):
                        st.markdown(para)
                    elif para.strip().startswith('1.') or para.strip().startswith('- '):
                        st.markdown(para)
                    else:
                        st.write(para)
                        st.write("")

                st.markdown("### 💡 Key Takeaways:")
                st.info(
                    "The content in this module will help you develop practical skills that you can apply immediately in your professional context.")

                st.markdown("### 📝 Module Quiz:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])

                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"module_{i}_question_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        options = q.get('options', [])
                        correct_answer = q.get('correct_answer', '')

                        # Display question only if not already answered
                        if question_id not in st.session_state.completed_questions:
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            
                            if options:
                                user_answer = st.radio(f"Select your answer for question {q_idx}:", 
                                                    options, 
                                                    key=f"{question_id}_radio")
                                
                                if st.button(f"Submit Answer", key=f"{question_id}_submit"):
                                    check_answer(question_id, user_answer, correct_answer)
                            else:
                                st.warning(f"No options available for question {q_idx}")
                        else:
                            st.success(f"✅ Question {q_idx} completed: {question_text}")

with tab2:
    st.header("❓ Employer Queries")
    
    if not st.session_state.employer_queries:
        st.info("No questions have been asked yet. Use the sidebar to submit questions about the course materials.")
    else:
        for i, query in enumerate(st.session_state.employer_queries):
            with st.expander(f"Question {i+1}: {query['question'][:80]}{'...' if len(query['question']) > 80 else ''}"):
                st.markdown("**Question:**")
                st.write(query['question'])
                st.markdown("**Answer:**")
                st.write(query['answer'] if query['answered'] else "This question has not been answered yet.")

with tab3:
    st.header("📑 Document Sources")
    
    if not st.session_state.extracted_texts:
        st.info("No documents have been uploaded yet. Use the sidebar to upload PDF files.")
    else:
        for i, doc in enumerate(st.session_state.extracted_texts):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                st.write(f"**File:** {doc['filename']}")
                preview_length = min(1000, len(doc['text']))
                st.write(f"**Preview:**\n{doc['text'][:preview_length]}{'...' if len(doc['text']) > preview_length else ''}")
