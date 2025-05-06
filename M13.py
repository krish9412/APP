import streamlit as st
import os
import tempfile
import json
import io
import uuid
import requests
import pdfplumber
import chromadb
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document as LangchainDocument
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI

# Create temp directory for database persistence
PERSIST_DIRECTORY = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Page Configuration
st.set_page_config(page_title="📚 Advanced Professional Learning Platform", layout="wide")

# Initializing sessions state variables
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
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    # Clear ChromaDB if it exists
    if st.session_state.db_initialized:
        try:
            import shutil
            shutil.rmtree(PERSIST_DIRECTORY)
            os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        except Exception as e:
            st.sidebar.error(f"Error clearing vector database: {e}")
    
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.db_initialized = False
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

# Create and initialize vector database from documents
def initialize_vector_db(documents, api_key):
    try:
        with st.spinner("🔄 Creating vector database from documents..."):
            # Create text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            # Prepare documents for langchain
            all_chunks = []
            for doc in documents:
                chunks = text_splitter.create_documents(
                    texts=[doc["text"]],
                    metadatas=[{"source": doc["filename"]}]
                )
                all_chunks.extend(chunks)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            
            # Create persistent ChromaDB vector store
            vectordb = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            vectordb.persist()
            
            return vectordb
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        st.error(str(e))
        return None

# Process uploaded files and add to session state
if uploaded_files and openai_api_key:
    # Clear previous uploads if list has changed
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        st.session_state.db_initialized = False
        
        # Extract text from each PDF and store in session state
        with st.spinner("Processing PDF files..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)
            
            # Initialize vector database after extracting all texts
            if st.session_state.extracted_texts:
                st.session_state.vector_store = initialize_vector_db(
                    st.session_state.extracted_texts, 
                    openai_api_key
                )
                if st.session_state.vector_store:
                    st.session_state.db_initialized = True
                    st.sidebar.success(f"✅ Vector database created successfully with {len(st.session_state.extracted_texts)} documents!")
                
        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")
else:
    st.info("📥 Please enter your OpenAI API key and upload PDF files to begin.")

# 🎯 GPT Model and Role selection
model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0)

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

# Enhanced RAG function using LangChain and vector search
def generate_rag_answer(question, documents=None, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        # Use vector database if available
        if st.session_state.db_initialized and st.session_state.vector_store:
            # Include course content for additional context if available
            course_context = ""
            if course_content:
                course_context = f"""
                Course Title: {course_content.get('course_title', '')}
                Course Description: {course_content.get('course_description', '')}
                
                Module Information:
                """
                for i, module in enumerate(course_content.get('modules', []), 1):
                    course_context += f"""
                    Module {i}: {module.get('title', '')}
                    Learning Objectives: {', '.join(module.get('learning_objectives', []))}
                    Content Summary: {module.get('content', '')[:200]}...
                    """
            
            # Create custom prompt template
            prompt_template = f"""
            You are an AI assistant for a professional learning platform. Answer the following question 
            based on the retrieved document content. Be specific, accurate, and helpful.
            
            Question: {{question}}
            
            {course_context}
            
            Context from documents: {{context}}
            
            Provide a comprehensive answer using information from the documents and course contents.
            If the question cannot be answered based on the provided information, say so politely.
            Reference specific documents when appropriate in your answer.
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["question", "context"]
            )
            
            # Setup LangChain retrieval QA chain
            chain_type_kwargs = {"prompt": PROMPT}
            llm = LangchainOpenAI(
                model_name=selected_model,
                temperature=0.5,
                openai_api_key=openai_api_key
            )
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )
            
            # Get answer from QA chain
            result = qa({"query": question})
            return result["result"]
        
        # Fallback to direct text search if vector DB not available
        elif documents:
            # Create a context from all document texts (with file attribution)
            combined_context = ""
            for i, doc in enumerate(documents[:3]):  # Limit to first 3 documents to avoid token issues
                context_chunk = doc["text"][:2000]  # Limit each doc to 2000 chars
                combined_context += f"\nDocument {i+1} ({doc['filename']}):\n{context_chunk}\n"
            
            # Include course content for additional context if available
            course_context = ""
            if course_content:
                course_context = f"""
                Course Title: {course_content.get('course_title', '')}
                Course Description: {course_content.get('course_description', '')}
                
                Module Information:
                """
                for i, module in enumerate(course_content.get('modules', []), 1):
                    course_context += f"""
                    Module {i}: {module.get('title', '')}
                    Learning Objectives: {', '.join(module.get('learning_objectives', []))}
                    Content Summary: {module.get('content', '')[:200]}...
                    """
            
            prompt = f"""
            You are an AI assistant for a professional learning platform. Answer the following question 
            based on the provided document content. Be specific, accurate, and helpful.
            
            Question: {question}
            
            Document Content: {combined_context}
            
            Course Information: {course_context}
            
            Provide a comprehensive answer using information from the documents and course contents.
            If the question cannot be answered based on the provided information, say so politely.
            Reference specific documents when appropriate in your answer.
            """
            
            # Create OpenAI client correctly
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            # Return generated answers
            return response.choices[0].message.content
        else:
            return "No documents or vector database available. Please process documents first."
            
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        # Generate proper answers automatically if documents are available
        answer = ""
        if st.session_state.extracted_texts:
            with st.spinner("Generating answer..."):
                answer = generate_rag_answer(
                    new_query, 
                    st.session_state.extracted_texts,
                    st.session_state.course_content if st.session_state.course_generated else None
                )
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
        # Add to completed questions set if not already there
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False

# Generate Progress Report
def generate_progress_report():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Training Progress Report")
    c.drawString(100, 730, f"User Role: {role}")
    c.drawString(100, 710, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    progress_percentage = (completed / total * 100) if total > 0 else 0
    c.drawString(100, 690, f"Progress: {completed}/{total} questions completed ({progress_percentage:.1f}%)")
    
    # Add course details
    if st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        c.drawString(100, 670, f"Course: {course.get('course_title', 'Professional Course')}")
        c.drawString(100, 650, "Module Completion:")
        
        y_pos = 630
        for i, module in enumerate(course.get('modules', []), 1):
            module_title = module.get('title', f'Module {i}')
            c.drawString(120, y_pos, f"Module {i}: {module_title}")
            y_pos -= 20
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Generate analytics data for course usage
def get_course_analytics():
    if not st.session_state.course_generated:
        return None
    
    # Create a dataframe with basic analytics
    data = {
        'Module': [],
        'Completion Rate': [],
        'Questions': [],
        'Completed': []
    }
    
    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    
    # Overall completion
    data['Module'].append('Overall')
    data['Completion Rate'].append(f"{(completed / total * 100) if total > 0 else 0:.1f}%")
    data['Questions'].append(total)
    data['Completed'].append(completed)
    
    # Module-specific completion
    modules = st.session_state.course_content.get('modules', [])
    for i, module in enumerate(modules, 1):
        module_questions = len(module.get('quiz', {}).get('questions', []))
        module_completed = sum(1 for qid in st.session_state.completed_questions if qid.startswith(f"module_{i}_"))
        
        data['Module'].append(f"Module {i}")
        data['Completion Rate'].append(f"{(module_completed / module_questions * 100) if module_questions > 0 else 0:.1f}%")
        data['Questions'].append(module_questions)
        data['Completed'].append(module_completed)
    
    return pd.DataFrame(data)

# Course Generation function
def generate_course():
    # Set generation flag to True when starting
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()  # Trigger rerun to show loading state

# Function to actually generate the course content
def perform_course_generation():
    try:
        # Use vector search for better document analysis if available
        if st.session_state.db_initialized and st.session_state.vector_store:
            # Get key concepts first using vector search
            key_concepts_query = "What are the key concepts, theories, and practical applications across all the documents?"
            key_concepts = generate_rag_answer(key_concepts_query)
            
            # Get document synthesis
            synthesis_query = "How do these documents relate to each other? What are the common themes and complementary concepts?"
            synthesis = generate_rag_answer(synthesis_query)
            
            professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
            
            prompt = f"""
            Design a comprehensive professional learning course based on the analyzed documents.
            
            Context: {professional_context}
            
            Key Concepts from Documents: {key_concepts}
            
            Document Synthesis and Relationships: {synthesis}
            
            Create an engaging, thorough and well-structured course by:
            1. Creating an inspiring course title that reflects the integrated knowledge from all documents
            2. Writing a detailed course description (at least 300 words) that explains how the course synthesizes information from multiple sources
            3. Developing 5-8 comprehensive modules that build upon each other in a logical sequence
            4. Providing 4-6 clear learning objectives for each module with specific examples and practical applications
            5. Creating detailed, well-explained content for each module (at least 500 words per module) including:
               - Real-world examples and case studies
               - Practical applications of concepts
               - Visual explanations where appropriate
               - Step-by-step guides for complex procedures
               - Comparative analysis when sources present different perspectives
            6. Including a quiz with 3-5 thought-provoking questions per module for better understanding
            
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
        else:
            # Fallback to direct document processing
            # Combine document texts for course generation with attribution
            combined_docs = ""
            for i, doc in enumerate(st.session_state.extracted_texts):
                doc_summary = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
                doc_summary += doc['text'][:3000]  # Limit each doc to avoid token limits
                combined_docs += doc_summary + "\n\n"
            
            professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
            
            # Get a document summary first
            summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
            document_summary = generate_rag_answer(summary_query, st.session_state.extracted_texts)
            
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
        
        try:
            # Create OpenAI client correctly
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Accessing the response content
            response_content = response.choices[0].message.content
            
            try:
                st.session_state.course_content = json.loads(response_content)
                st.session_state.course_generated = True
                
                # Count total questions for progress tracking
                total_questions = 0
                for module in st.session_state.course_content.get("modules", []):
                    quiz = module.get("quiz", {})
                    total_questions += len(quiz.get("questions", []))
                st.session_state.total_questions = total_questions
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {e}")
                st.text(response_content)
        
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
            st.error("Please check your API key and model selection.")
            
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Always reset the generation flag when done
    st.session_state.is_generating = False

# Main contents area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources", "📊 Analytics"])

# Check if we're in the middle of generating a course and need to continue
if st.session_state.is_generating:
    with st.spinner("Generating your personalized course from multiple documents..."):
        # Reset completed questions when generating a new course
        st.session_state.completed_questions = set()
        perform_course_generation()
        st.success("✅ Your Comprehensive Course is Ready!")
        st.rerun()  # Refresh the UI after completion

with tab1:
    # Display Course Content
    if st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        
        # Course Header with appreciation
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.markdown(f"*Specially designed for {role}s focusing on {', '.join(learning_focus)}*")
        st.write(course.get('course_description', 'A structured course to enhance your skills.'))
        
        # Tracking the Progress
        completed = len(st.session_state.completed_questions)
        total = st.session_state.total_questions
        progress_percentage = (completed / total * 100) if total > 0 else 0
        
        st.progress(progress_percentage / 100)
        st.write(f"**Progress:** {completed}/{total} questions completed ({progress_percentage:.1f}%)")
        # Add Download Report button
        st.download_button("📥 Download Progress Report", generate_progress_report(), "progress_report.pdf")
        
        st.markdown("---")
        st.subheader("📋 Course Overview")
        
        # Safely access module titles
        modules = course.get("modules", [])
        if modules:
            modules_list = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, module_title in enumerate(modules_list, 1):
                st.write(f"**Module {i}:** {module_title}")
        else:
            st.warning("No modules were found in the course content.")
        
        st.markdown("---")
        
        # Detailed Module Contents with improved formatting
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                # Module Learning Objectives
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives specified.")
                
                # Module Content with better readability
                st.markdown("### 📖 Module Content:")
                module_content = module.get('content', 'No content available for this module.')
                
                # Split the content into paragraphs and add proper formatting
                paragraphs = module_content.split('\n\n')
                for para in paragraphs:
                    if para.strip().startswith('#'):
                        # Handle markdown headers
                        st.markdown(para)
                    elif para.strip().startswith('*') and para.strip().endswith('*'):
                        # Handle emphasized text
                        st.markdown(para)
                    elif para.strip().startswith('1.') or para.strip().startswith('- '):
                        # Handle lists
                        st.markdown(para)
                    else:
                        # Regular paragraphs
                        st.write(para)
                        st.write("")  # Add spacing between paragraphs
                
                # Key Takeaways section
                st.markdown("### 💡 Key Takeaways:")
                st.info("The content in this module will help you develop practical skills that you can apply immediately in your professional context.")
                
                # Module Quiz with improved UI
                st.markdown("### 📝 Module Quiz:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"module_{i}_question_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        
                        # Create quiz question container
                        quiz_container = st.container()
                        with quiz_container:
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            
                            options = q.get('options', [])
                            if options:
                                # Create a unique key for each radio button
                                option_key = f"quiz_{i}_{q_idx}"
                                user_answer = st.radio("Select your answer:", options, key=option_key)
                                
                                # Create a unique key for each submit button
                                submit_key = f"submit_{i}_{q_idx}"
                                
                                # Show completion status for this question
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
        # Welcome screen when no course is generated yet
        st.title("Welcome to Advanced Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning system
        
        Upload multiple PDF documents, and I'll create a comprehensive, integrated learning course just for you!
        
        ### How it works:
        1. Enter your OpenAI API key in the sidebar
        2. Select your professional role and learning focus
        3. Upload multiple PDF documents related to your area of interest
        4. Click "Generate Course" to create your personalized learning journey that combines insights from all documents
        
        This enhanced version uses:
        - **ChromaDB** for vector search and semantic retrieval
        - **LangChain** for document processing and advanced RAG
        - **PDFPlumber** for precise PDF text extraction
        - **ReportLab** for professional PDF report generation
        
        Get ready to enhance your skills and accelerate your professional growth with our advanced learning platform!
        """
        
        # Generate Course Button - only if not currently generating
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
    
    With our enhanced vector search capabilities, we can provide more accurate and relevant answers!
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
                    # Generate answer on-demand if not already answered
                    if st.session_state.extracted_texts:
                        try:
                            answer = generate_rag_answer(
                                query['question'], 
                                st.session_state.extracted_texts,
                                st.session_state.course_content if st.session_state.course_generated else None
                            )
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
        
        # Add a search box to search across all documents
        search_query = st.text_input("🔍 Search across all documents")
        if search_query and len(search_query) > 2:
            st.subheader("Search Results")
            with st.spinner("Searching documents..."):
                # Use vector search if available
                if st.session_state.db_initialized and st.session_state.vector_store:
                    # Create a search query
                    search_results = st.session_state.vector_store.similarity_search_with_score(search_query, k=5)
                    
                    if search_results:
                        for i, (doc, score) in enumerate(search_results):
                            relevance = 100 * (1 - score/2)  # Convert distance to relevance percentage
                            st.write(f"**Result {i+1}** (Relevance: {relevance:.1f}%)")
                            st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"**Content:** {doc.page_content}")
                            st.markdown("---")
                    else:
                        st.info("No matching content found in the documents.")
                else:
                    # Basic text search
                    results_found = False
                    for doc in st.session_state.extracted_texts:
                        if search_query.lower() in doc['text'].lower():
                            st.write(f"**Found in:** {doc['filename']}")
                            # Show the context around the search term
                            text = doc['text'].lower()
                            start_idx = text.find(search_query.lower())
                            if start_idx != -1:
                                context_start = max(0, start_idx - 100)
                                context_end = min(len(text), start_idx + len(search_query) + 100)
                                context = doc['text'][context_start:context_end]
                                st.text_area("Context:", value=f"...{context}...", height=100, disabled=True)
                                results_found = True
                    
                    if not results_found:
                        st.info("No matching content found in the documents.")
        
        # Show document list
        for i, doc in enumerate(st.session_state.extracted_texts):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                # Display document preview (first 1000 characters)
                preview_text = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                st.markdown("### Document Preview:")
                st.text_area("Content Preview:", value=preview_text, height=300, disabled=True)
                
                # Add document summary using AI
                if st.button(f"Generate Summary for {doc['filename']}", key=f"sum_{i}"):
                    with st.spinner("Generating document summary..."):
                        summary_query = f"Create a comprehensive summary of this document highlighting key concepts, theories, and practical applications:"
                        summary = generate_rag_answer(summary_query, [doc])
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)

with tab4:
    st.title("📊 Analytics Dashboard")
    
    if not st.session_state.course_generated:
        st.info("Generate a course first to view analytics.")
    else:
        # Display course analytics
        st.subheader("Course Progress Analytics")
        
        # Get analytics data
        analytics_data = get_course_analytics()
        if analytics_data is not None:
            # Display overall stats
            col1, col2, col3 = st.columns(3)
            with col1:
                overall_completion = analytics_data.iloc[0]['Completion Rate']
                st.metric("Overall Completion", overall_completion)
            
            with col2:
                completed = analytics_data.iloc[0]['Completed']
                total = analytics_data.iloc[0]['Questions']
                st.metric("Questions Completed", f"{completed}/{total}")
            
            with col3:
                # Calculate time spent (estimated)
                # In a real app, you'd track actual time
                estimated_time = completed * 5  # Assume 5 minutes per question
                st.metric("Estimated Time Spent", f"{estimated_time} minutes")
            
            # Display analytics table
            st.subheader("Module-by-Module Progress")
            st.dataframe(analytics_data)
            
            # Add visualization
            st.subheader("Completion by Module")
            
            # Prepare data for chart
            module_data = analytics_data.iloc[1:].copy()  # Skip overall
            module_data['Completion'] = module_data['Completion Rate'].str.rstrip('%').astype(float)
            
            # Create a horizontal bar chart
            st.bar_chart(module_data.set_index('Module')['Completion'])
            
            # Add learning pattern analysis
            st.subheader("Learning Pattern Analysis")
            st.write("""
            Based on your interaction patterns, we recommend:
            
            1. **Focus on challenging modules**: Spend more time on modules with lower completion rates
            2. **Regular review**: Schedule review sessions for completed modules to reinforce learning
            3. **Practice application**: Try applying concepts from completed modules in real-world scenarios
            """)
            
            # Add export option
            st.download_button(
                "📥 Export Analytics Data (CSV)",
                analytics_data.to_csv(index=False).encode('utf-8'),
                "course_analytics.csv",
                "text/csv",
                key='download-csv'
            )

# Add footer
st.markdown("---")
st.markdown("📚 **Advanced Professional Learning Platform** | Powered by LangChain & ChromaDB")