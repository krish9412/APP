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

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS  # Using FAISS instead of Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Page Configuration
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Initializing session state variables
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
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.vector_store = None
    st.rerun()

# 🔐 OpenAI API Key Inputs
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
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

# Process uploaded files and add to session state
if uploaded_files and openai_api_key:
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        documents = []
        
        with st.spinner("Processing PDF files..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)
                    documents.append(Document(page_content=extracted_text, metadata={"filename": pdf_file.name}))
                    
        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")
            
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(documents)
                
                embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                # Use FAISS instead of Chroma for better compatibility
                st.session_state.vector_store = FAISS.from_documents(
                    documents=split_docs,
                    embedding=embeddings
                )
            except Exception as e:
                st.error(f"Error initializing vector store: {e}")
else:
    st.info("📥 Please enter your OpenAI API key and upload PDF files to begin.")

# 🎯 GPT Model and Role selection
model_options = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0 if "gpt-4.1-nano" in model_options else 1)

role_options = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
role = st.sidebar.selectbox("Select Your Role", role_options)

learning_focus_options = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
learning_focus = st.sidebar.multiselect("Select Learning Focus", learning_focus_options)

# Display uploaded files in sidebar
if st.session_state.uploaded_file_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for i, filename in enumerate(st.session_state.uploaded_file_names):
        st.sidebar.text(f"{i+1}. {filename}")

# Enhanced RAG function using LangChain with FAISS
def generate_rag_answer(question, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        if not st.session_state.vector_store:
            return "Vector store not initialized. Please process documents first."
            
        # Retrieve relevant documents
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(question)
        
        # Construct the context from retrieved documents
        context = ""
        for doc in retrieved_docs:
            context += f"\nDocument: {doc.metadata.get('filename', 'Unknown')}\nContent: {doc.page_content[:500]}...\n"
        
        # Construct the course context if available
        course_context = ""
        if course_content:
            course_context = f"""
            Course Title: {course_content.get('course_title', 'N/A')}
            Course Description: {course_content.get('course_description', 'No description available.')}
            
            Module Information:
            """
            for i, module in enumerate(course_content.get('modules', []), 1):
                course_context += f"""
                Module {i}: {module.get('title', 'Untitled Module')}
                Learning Objectives: {', '.join(module.get('learning_objectives', ['None specified']))}
                Content Summary: {module.get('content', 'No content available.')[:200]}...
                """
        else:
            course_context = "No course content available."
        
        # Construct the prompt manually
        prompt = f"""
        You are an AI assistant for a professional learning platform. Answer the following question 
        based on the provided document content and course information. Be specific, accurate, and helpful.
        
        Question: {question}
        
        Document Content: {context}
        
        Course Information: {course_context}
        
        Provide a comprehensive answer using information from the documents and course contents.
        If the question cannot be answered based on the provided information, say so politely.
        Reference specific documents when appropriate in your answer.
        """
        
        # Initialize the LLM
        llm = ChatOpenAI(api_key=openai_api_key, model=selected_model, temperature=0.5)
        
        # Call the LLM directly with the constructed prompt
        response = llm.invoke(prompt)
        answer = response.content
        
        # Append references to the answer
        if retrieved_docs:
            answer += "\n\n**References:**\n"
            for doc in retrieved_docs:
                filename = doc.metadata.get("filename", "Unknown")
                answer += f"- Document: {filename}\n"
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        answer = ""
        if st.session_state.vector_store:
            with st.spinner("Generating answer..."):
                try:
                    current_course = st.session_state.course_content if hasattr(st.session_state, 'course_generated') and st.session_state.course_generated else None
                    answer = generate_rag_answer(new_query, current_course)
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
        else:
            answer = "Please upload and process documents first to enable question answering."
        
        st.session_state.employer_queries.append({
            "question": new_query,
            "answer": answer,
            "answered": bool(answer)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Functions to check answer and update progress
def check_answer(question_id, user_answer, correct_answer):
    if user_answer == correct_answer:
        st.success("🎉 Correct! Well done!")
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False

# Generate Progress Report
def generate_progress_report():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "Professional Learning Platform")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 70, "Training Progress Report")
    c.line(50, height - 80, width - 50, height - 80)

    c.setFont("Helvetica", 12)
    y_position = height - 110

    c.drawString(50, y_position, f"User Role: {role}")
    y_position -= 20
    c.drawString(50, y_position, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    y_position -= 20

    if hasattr(st.session_state, 'course_content') and st.session_state.course_content:
        course_title = st.session_state.course_content.get('course_title', 'N/A')
        c.drawString(50, y_position, f"Course: {course_title}")
        y_position -= 20
    if learning_focus:
        c.drawString(50, y_position, f"Learning Focus: {', '.join(learning_focus)}")
        y_position -= 20

    y_position -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Progress Overview:")
    c.setFont("Helvetica", 12)
    y_position -= 20
    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    progress_percentage = (completed / total * 100) if total > 0 else 0
    c.drawString(50, y_position, f"Questions Completed: {completed}/{total}")
    y_position -= 20
    c.drawString(50, y_position, f"Progress Percentage: {progress_percentage:.1f}%")
    y_position -= 20

    y_position -= 10
    if progress_percentage >= 75:
        summary = "Excellent progress! You're almost done!"
    elif progress_percentage >= 50:
        summary = "Great job! You're more than halfway there!"
    elif progress_percentage > 0:
        summary = "Good start! Keep going to complete more modules!"
    else:
        summary = "Let's get started! Answer some quiz questions to track your progress."
    c.drawString(50, y_position, f"Summary: {summary}")
    y_position -= 20

    if st.session_state.completed_questions:
        y_position -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Completed Questions:")
        c.setFont("Helvetica", 12)
        y_position -= 20
        for qid in sorted(st.session_state.completed_questions):
            try:
                module_num = qid.split('_')[1]
                question_num = qid.split('_')[3]
                c.drawString(50, y_position, f"Module {module_num}, Question {question_num}")
                y_position -= 15
                if y_position < 50:
                    c.showPage()
                    y_position = height - 50
            except IndexError:
                continue

    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 30, f"Generated by Professional Learning Platform on {datetime.now().strftime('%Y-%m-%d')}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Course Generation function
def generate_course():
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()

# Function to actually generate the course content
def perform_course_generation():
    try:
        combined_docs = ""
        for i, doc in enumerate(st.session_state.extracted_texts):
            doc_summary = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
            doc_summary += doc['text'][:3000]
            combined_docs += doc_summary + "\n\n"
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
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
        
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=selected_model,
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
        st.error(f"Error: {e}")
    
    st.session_state.is_generating = False

# Main contents area with tabs
tab1, tab2, tab3 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

if st.session_state.is_generating:
    with st.spinner("Generating your personalized course from multiple documents..."):
        st.session_state.completed_questions = set()
        perform_course_generation()
        st.success("✅ Your Comprehensive Course is Ready!")
        st.rerun()

with tab1:
    if hasattr(st.session_state, 'course_generated') and st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.markdown(f"*Specially designed for {role}s focusing on {', '.join(learning_focus)}*")
        st.write(course.get('course_description', 'A structured course to enhance your skills.'))
        
        completed = len(st.session_state.completed_questions)
        total = st.session_state.total_questions
        progress_percentage = (completed / total * 100) if total > 0 else 0
        
        st.progress(progress_percentage / 100)
        st.write(f"**Progress:** {completed}/{total} questions completed ({progress_percentage:.1f}%)")
        st.download_button("📥 Download Progress Report", generate_progress_report(), "progress_report.pdf")
        
        st.markdown("---")
        st.subheader("📋 Course Overview")
        
        modules = course.get("modules", [])
        if modules:
            modules_list = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, module_title in enumerate(modules_list, 1):
                st.write(f"**Module {i}:** {module_title}")
        else:
            st.warning("No modules were found in the course content.")
        
        st.markdown("---")
        
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
                st.info("The content in this module will help you develop practical skills that you can apply immediately in your professional context.")
                
                st.markdown("### 📝 Module Quiz:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"module_{i}_question_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        
                        quiz_container = st.container()
                        with quiz_container:
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            options = q.get('options', [])
                            if options:
                                option_key = f"quiz_{i}_{q_idx}"
                                user_answer = st.radio("Select your answer:", options, key=option_key)
                                submit_key = f"submit_{i}_{q_idx}"
                                if question_id in st.session_state.completed_questions:
                                    st.success("✓ Question completed")
                                else:
                                    if st.button(f"Check Answer", key=submit_key):
                                        correct_answer = q.get('correct_answer', '')
                                        check_answer(question_id, user_answer, correct_answer)
                            else:
                                st.write("No options available for this question.")
                        
                        st.markdown("---")
                else:
                    st.write("No quiz questions available for this module.")

    else:
        st.title("Welcome to Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning system
        
        Upload multiple PDF documents, and I'll create a comprehensive, integrated learning course just for you!
        
        ### How it works:
        1. Enter your OpenAI API key in the sidebar
        2. Select your professional role and learning focus
        3. Upload multiple PDF documents related to your area of interest
        4. Click "Generate Course" to create your personalized learning journey that combines insights from all documents
        
        Get ready to enhance your skills and accelerate your professional growth!
        """)
        
        if st.session_state.extracted_texts and openai_api_key and not st.session_state.is_generating:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate My Course", use_container_width=True):
                    generate_course()
        elif st.session_state.is_generating:
            st.info("Generating your personalized course... Please wait.")

with tab2:
    st.title("💬 Employer Queries")
    st.markdown("""
    This section allows employers to ask questions and get AI-generated answers about the course content or related topics.
    Submit your questions in the sidebar, and our AI will automatically generate answers based on the uploaded documents.
    """)
    
    if not st.session_state.employer_queries:
        st.info("No questions have been submitted yet. Add a question in the sidebar to get started.")
    else:
        for i, query in enumerate(st.session_state.employer_queries):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                if query['answered']:
                    st.write(f"**Answer:** {query['answer']}")
                else:
                    st.info("Generating answer...")
                    if st.session_state.vector_store:
                        try:
                            current_course = st.session_state.course_content if hasattr(st.session_state, 'course_generated') and st.session_state.course_generated else None
                            answer = generate_rag_answer(query['question'], current_course)
                            st.session_state.employer_queries[i]['answer'] = answer
                            st.session_state.employer_queries[i]['answered'] = True
                            st.rerun()
                        except Exception as e:
                            error_msg = f"Error generating answer: {str(e)}. Please try resetting the application."
                            st.error(error_msg)
                            st.session_state.employer_queries[i]['answer'] = error_msg
                            st.session_state.employer_queries[i]['answered'] = True
                    else:
                        st.warning("No documents uploaded yet. Please upload documents to generate answers.")

with tab3:
    st.title("📑 Document Sources")
    
    if not st.session_state.extracted_texts:
        st.info("No documents have been uploaded yet. Please upload PDF files in the sidebar to see their content here.")
    else:
        st.write(f"**{len(st.session_state.extracted_texts)} documents uploaded:**")
        
        for i, doc in enumerate(st.session_state.extracted_texts):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                preview_text = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                st.markdown("### Document Preview:")
                st.text_area("Content Preview:", value=preview_text, height=300, disabled=True)
                
                if st.button(f"Generate Summary for {doc['filename']}", key=f"sum_{i}"):
                    with st.spinner("Generating document summary..."):
                        summary_query = f"Create a comprehensive summary of the document '{doc['filename']}' highlighting key concepts, theories, and practical applications:"
                        summary = generate_rag_answer(summary_query)
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
