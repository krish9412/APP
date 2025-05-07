import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
import chromadb
import time
from datetime import datetime
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Create a temporary directory for ChromaDB persistence
TEMP_DIR = tempfile.mkdtemp()
CHROMA_DB_DIR = os.path.join(TEMP_DIR, "chroma_db")

# Page Configuration
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

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
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    # Clean up ChromaDB files if they exist
    if os.path.exists(CHROMA_DB_DIR):
        try:
            import shutil
            shutil.rmtree(CHROMA_DB_DIR)
        except:
            pass
    
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.rerun()

# 🔐 OpenAI API Key Inputs
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to extract text from PDF with page information
def extract_pdf_text(pdf_file):
    try:
        pdf_file.seek(0)
        all_text = []
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():  # Only add non-empty pages
                    all_text.append({
                        "page": page_num + 1,
                        "text": page_text
                    })
        return all_text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return []

# Function to create text chunks for vector storage
def create_text_chunks(texts, filename):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    documents = []
    for page_data in texts:
        if page_data["text"].strip():
            chunk = Document(
                page_content=page_data["text"],
                metadata={
                    "source": filename,
                    "page": page_data["page"]
                }
            )
            documents.append(chunk)
    
    return text_splitter.split_documents(documents)

# Initialize ChromaDB and Langchain components
def initialize_vector_db(documents, api_key):
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Create Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        
        # Create QA chain
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4o-mini",  # Default to the cheapest model
            openai_api_key=api_key
        )
        
        # Custom prompt template for better answers
        template = """
        You are an AI assistant for a professional learning platform. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say you don't know. DO NOT make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer in a comprehensive, professional manner:
        """
        
        QA_PROMPT = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )
        
        # Update session state
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        
        return True
    
    except Exception as e:
        st.error(f"Error initializing vector database: {str(e)}")
        return False

# Process uploaded files and add to session state
if uploaded_files and openai_api_key:
    # Clear previous uploads if list has changed
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        
        # Extract text from each PDF and store in session state
        with st.spinner("Processing PDF files..."):
            all_chunks = []
            
            for pdf_file in uploaded_files:
                extracted_pages = extract_pdf_text(pdf_file)
                
                if extracted_pages:
                    # Combine all text for the simple extracted_texts
                    combined_text = "\n\n".join([page["text"] for page in extracted_pages])
                    
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": combined_text
                    })
                    
                    # Create chunks for vector DB
                    chunks = create_text_chunks(extracted_pages, pdf_file.name)
                    all_chunks.extend(chunks)
                    
                    # Store original file
                    st.session_state.uploaded_files.append(pdf_file)
            
            # Initialize vector database with all document chunks
            if all_chunks:
                with st.spinner("Building knowledge base..."):
                    success = initialize_vector_db(all_chunks, openai_api_key)
                    if success:
                        st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed and indexed!")
                    else:
                        st.sidebar.error("Failed to build knowledge base. Please check API key and try again.")
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

# Enhanced RAG function using Langchain
def generate_rag_answer(question, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        if st.session_state.qa_chain:
            # Add course content for more context if available
            context_prefix = ""
            if course_content:
                context_prefix = f"""
                Consider this additional course information in your answer:
                
                Course Title: {course_content.get('course_title', '')}
                Course Description: {course_content.get('course_description', '')}
                """
            
            # Full question with context
            full_question = f"{context_prefix}\n\n{question}"
            
            # Get answer from QA chain
            result = st.session_state.qa_chain({"query": full_question})
            answer = result.get("result", "")
            
            source_docs = result.get("source_documents", [])
            sources_text = ""
            
            # Add source information
            if source_docs:
                unique_sources = set()
                for doc in source_docs:
                    source = f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})"
                    unique_sources.add(source)
                
                sources_text = "\n\nSources:\n" + "\n".join([f"- {source}" for source in unique_sources])
            
            return answer + sources_text
        else:
            # Fallback to simple context-based answers
            if not st.session_state.extracted_texts:
                return "Document text is not available. Please process documents first."
                
            # Create a simple context from all document texts (with file attribution)
            combined_context = ""
            for i, doc in enumerate(st.session_state.extracted_texts[:3]):  # Limit to first 3 documents
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
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Enhanced Progress Report with ReportLab
def generate_progress_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=8
    )
    
    # Add report title
    elements.append(Paragraph("Professional Learning Progress Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Add user information
    elements.append(Paragraph(f"User Role: {role}", normal_style))
    elements.append(Paragraph(f"Learning Focus: {', '.join(learning_focus)}", normal_style))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Add progress metrics
    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    progress_percentage = (completed / total * 100) if total > 0 else 0
    
    elements.append(Paragraph("Progress Summary", heading_style))
    elements.append(Paragraph(f"Questions Completed: {completed}/{total}", normal_style))
    elements.append(Paragraph(f"Completion Rate: {progress_percentage:.1f}%", normal_style))
    elements.append(Spacer(1, 12))
    
    # Add course content summary if available
    if st.session_state.course_content:
        course = st.session_state.course_content
        elements.append(Paragraph("Course Summary", heading_style))
        elements.append(Paragraph(f"Course Title: {course.get('course_title', '')}", normal_style))
        
        # Create a table for modules
        if course.get("modules"):
            elements.append(Paragraph("Module Progress", heading_style))
            
            table_data = [["Module", "Completion"]]
            
            for i, module in enumerate(course.get("modules", []), 1):
                module_title = module.get('title', f'Module {i}')
                
                # Calculate module completion
                module_questions = 0
                module_completed = 0
                
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                module_questions = len(questions)
                for q_idx, _ in enumerate(questions, 1):
                    question_id = f"module_{i}_question_{q_idx}"
                    if question_id in st.session_state.completed_questions:
                        module_completed += 1
                
                module_progress = (module_completed / module_questions * 100) if module_questions > 0 else 0
                table_data.append([f"Module {i}: {module_title}", f"{module_completed}/{module_questions} ({module_progress:.1f}%)"])
            
            # Create the table
            table = Table(table_data, colWidths=[doc.width*0.7, doc.width*0.3])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(table)
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        # Generate proper answers automatically if vector store is available
        answer = ""
        if st.session_state.qa_chain:
            with st.spinner("Generating answer..."):
                answer = generate_rag_answer(
                    new_query, 
                    st.session_state.course_content if st.session_state.course_generated else None
                )
        else:
            answer = "Please upload and process documents first to enable question answering."
        
        st.session_state.employer_queries.append({
            "question": new_query,
            "answer": answer,
            "answered": bool(answer),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

# Course Generation function
def generate_course():
    # Set generation flag to True when starting
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()  # Trigger rerun to show loading state

# Function to actually generate the course content
def perform_course_generation():
    try:
        # Use vector store for better document retrieval
        summaries = []
        
        # Generate summaries for each document first
        for i, doc in enumerate(st.session_state.extracted_texts):
            summary_query = f"Create a comprehensive summary of the document titled '{doc['filename']}' highlighting key concepts, theories, and practical applications."
            doc_summary = generate_rag_answer(summary_query)
            summaries.append({
                "filename": doc['filename'],
                "summary": doc_summary
            })
        
        # Combine document summaries for course generation
        combined_summaries = ""
        for i, summary in enumerate(summaries, 1):
            combined_summaries += f"\n--- DOCUMENT {i}: {summary['filename']} ---\n"
            combined_summaries += summary['summary'] + "\n\n"
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
        # Get an integrated concept map of the documents
        concept_map_query = "Create a concept map showing how all these documents relate to each other. Identify common themes, complementary concepts, and areas where the documents provide different perspectives on the same topic."
        concept_map = generate_rag_answer(concept_map_query)
        
        prompt = f"""
        Design a comprehensive professional learning course based on multiple documents that have been analyzed.
        Context: {professional_context}
        
        Concept Map of Documents: {concept_map}
        
        Document Summaries: {combined_summaries}
        
        Create an engaging, thorough and well-structured course by:
        1. Analyzing all provided document summaries and identifying common themes, complementary concepts, and unique insights
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
            # Update the model based on user selection
            if st.session_state.qa_chain:
                # Create a new ChatOpenAI with the selected model
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
            else:
                st.error("Vector database not initialized. Please try resetting the application and uploading documents again.")
        
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
            st.error("Please check your API key and model selection.")
            
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Always reset the generation flag when done
    st.session_state.is_generating = False

# Main contents area with tabs
tab1, tab2, tab3 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

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
        st.download_button(
            "📥 Download Progress Report", 
            generate_progress_report(), 
            "progress_report.pdf", 
            "application/pdf",
            use_container_width=True
        )
        
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
        
        # Generate Course Button - only if not currently generating
        if st.session_state.extracted_texts and openai_api_key and not st.session_state.is_generating:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate My Course", use_container_width=True):
                    generate_course()
        elif st.session_state.is_generating:
            st.info("Generating your personalized course... Please wait.")
