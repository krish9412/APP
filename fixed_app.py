import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from openai import OpenAI 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="📚 Enhanced Professional Learning Platform", layout="wide")

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
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'langchain_qa_chain' not in st.session_state:
    st.session_state.langchain_qa_chain = None
if 'chroma_client' not in st.session_state:
    # Initialize ChromaDB client with persistent storage
    persist_directory = os.path.join(tempfile.gettempdir(), 'chromadb_' + st.session_state.session_id)
    st.session_state.chroma_client = chromadb.Client(Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    # Clean up the persistent ChromaDB directory if it exists
    persist_directory = os.path.join(tempfile.gettempdir(), 'chromadb_' + st.session_state.session_id)
    if os.path.exists(persist_directory):
        import shutil
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            st.sidebar.error(f"Error cleaning up ChromaDB: {e}")
    
    # Reset session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize critical session state variables
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    
    # Re-initialize ChromaDB client with new persistent storage
    persist_directory = os.path.join(tempfile.gettempdir(), 'chromadb_' + st.session_state.session_id)
    st.session_state.chroma_client = chromadb.Client(Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))
    
    st.rerun()

# 🔐 OpenAI API Key Inputs
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to save uploaded PDF to a temporary file
def save_temp_pdf(uploaded_file):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

# Function to extract text using pdfplumber
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

# Setup LangChain and ChromaDB for advanced retrieval
def setup_langchain_retrieval(uploaded_files, api_key):
    try:
        with st.spinner("Setting up advanced document retrieval system..."):
            # Initialize OpenAI embeddings
            embeddings = OpenAIEmbeddings(api_key=api_key)
            
            # Create a text splitter for chunking documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            # Process all documents
            all_docs = []
            for pdf_file in uploaded_files:
                # Save the uploaded PDF to a temporary file
                temp_path = save_temp_pdf(pdf_file)
                
                # Use LangChain's PyPDFLoader to load the document
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                
                # Add source document name to metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_file.name
                
                # Split the documents into chunks
                docs = text_splitter.split_documents(documents)
                all_docs.extend(docs)
            
            # Create a persistent Chroma vector store
            persist_directory = os.path.join(tempfile.gettempdir(), 'chroma_db_' + st.session_state.session_id)
            vector_db = Chroma.from_documents(
                documents=all_docs,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            
            # Create a retriever
            retriever = vector_db.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={"k": 5, "fetch_k": 8}  # Retrieve 5 docs, consider 8 for diversity
            )
            
            # Configure custom prompt template for RAG
            template = """
            You are an AI assistant for a professional learning platform. Use the following context to answer the question.
            
            Context: {context}
            
            Question: {question}
            
            If the answer cannot be found in the context, acknowledge that and provide your best response
            based on general knowledge. Always cite specific document sources from the retrieved context when possible.
            
            Conversation History: {chat_history}
            
            Answer:
            """
            
            prompt = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=template
            )
            
            # Setup memory for conversation history
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=False
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=0.7,
                    model_name=selected_model,
                    api_key=api_key
                ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "memory": memory
                }
            )
            
            return vector_db, retriever, qa_chain
            
    except Exception as e:
        st.error(f"Error setting up LangChain components: {str(e)}")
        return None, None, None

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
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)
            
            # Set up LangChain and ChromaDB components
            if len(st.session_state.extracted_texts) > 0:
                vector_db, retriever, qa_chain = setup_langchain_retrieval(uploaded_files, openai_api_key)
                st.session_state.vector_db = vector_db
                st.session_state.retriever = retriever
                st.session_state.langchain_qa_chain = qa_chain
                    
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

# Enhanced RAG function using LangChain and ChromaDB
def generate_rag_answer(question, documents=None, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        # Use LangChain QA chain if available
        if st.session_state.langchain_qa_chain is not None:
            result = st.session_state.langchain_qa_chain({"query": question})
            answer = result["result"]
            
            # Enhance answer with course content if available
            if course_content:
                # Create OpenAI client
                client = OpenAI(api_key=openai_api_key)
                
                # Create a prompt to enhance the answer with course content
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
                
                enhancement_prompt = f"""
                I have an answer to a question but want to enhance it with additional course content.
                
                Question: {question}
                
                Initial Answer: {answer}
                
                Course Content: {course_context}
                
                Please enrich the answer with relevant information from the course content while 
                maintaining the accuracy and key points from the initial answer. Make connections 
                between the question, initial answer, and course modules where appropriate.
                """
                
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": enhancement_prompt}],
                    temperature=0.4
                )
                
                # Return the enhanced answer
                return response.choices[0].message.content
            else:
                # Return the original answer if no course content is available
                return answer
                
        # Fallback to direct OpenAI if LangChain setup failed
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
            return "Document text is not available. Please process documents first."
            
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
            with st.spinner("Generating answer using advanced retrieval..."):
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
    c.drawString(100, 690, f"Progress: {completed}/{total} questions completed ({completed/total*100:.1f}%)")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Course Generation function
def generate_course():
    # Set generation flag to True when starting
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()  # Trigger rerun to show loading state

# Function to actually generate the course content
def perform_course_generation():
    try:
        # Get document summaries using LangChain if available
        document_summary = ""
        if st.session_state.langchain_qa_chain:
            summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
            result = st.session_state.langchain_qa_chain({"query": summary_query})
            document_summary = result["result"]
        else:
            # Combine document texts for course generation with attribution
            combined_docs = ""
            for i, doc in enumerate(st.session_state.extracted_texts):
                doc_summary = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
                doc_summary += doc['text'][:3000]  # Limit each doc to avoid token limits
                combined_docs += doc_summary + "\n\n"
            
            # Get a document summary first with direct API call
            summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
            document_summary = generate_rag_answer(summary_query, st.session_state.extracted_texts)
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
        # Retrieve top chunks from vector store to use as context
        vector_context = ""
        if st.session_state.retriever:
            context_docs = st.session_state.retriever.get_relevant_documents(
                "key concepts theories practical applications professional development"
            )
            for i, doc in enumerate(context_docs[:5]):
                vector_context += f"\nChunk {i+1} from {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}\n"
        
        # Combined context from both methods
        combined_context = f"Document Summary: {document_summary}\n\nVectorDB Retrieved Content: {vector_context}"
        
        prompt = f"""
        Design a comprehensive professional learning course based on the multiple documents provided.
        Context: {professional_context}
        
        Document Information: {combined_context[:7000]}
        
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
tab1, tab2, tab3, tab4 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources", "🔍 Interactive Search"])

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
        st.title("Welcome to Enhanced Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning system
        
        Upload multiple PDF documents, and I'll create a comprehensive, integrated learning course just for you!
        
        ### What's new in this version:
        - ChromaDB for advanced vector storage of documents
        - LangChain for improved document processing and retrieval
        - Semantic search capabilities across all documents
        - Better question answering with advanced RAG techniques
        
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

with tab2:
    st.title("💬 Employer Queries")
    st.markdown("""
    This section allows employers to ask questions and get AI-generated answers about the course content or related topics.
    Submit your questions in the sidebar, and our advanced AI will automatically generate detailed answers based on the uploaded documents using semantic search.
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
    st.title("🔍 Interactive Document Search")
    
    if not st.session_state.vector_db or not st.session_state.retriever:
        st.warning("Please upload documents to enable interactive search capabilities.")
    else:
        st.markdown("""
        This new search feature uses semantic search to find the most relevant content across your uploaded documents.
        It goes beyond simple keyword matching to understand the meaning behind your query.
        """)
        
        search_query = st.text_input("🔎 Search across your documents:", placeholder="Enter your search query here...")
        
        if search_query:
            with st.spinner("Searching documents..."):
                try:
                    # Get relevant documents from the retriever
                    docs = st.session_state.retriever.get_relevant_documents(search_query)
                    
                    st.success(f"Found {len(docs)} relevant results!")
                    
                    for i, doc in enumerate(docs):
                        with st.expander(f"Result {i+1} - From: {doc.metadata.get('source', 'Unknown document')}"):
                            st.markdown("### Content:")
                            st.write(doc.page_content)
                            st.markdown("---")
                            st.markdown("### Metadata:")
                            st.json(doc.metadata)
                    
                    # Generate a summarized answer to the search query
                    with st.spinner("Generating concise answer to your query..."):
                        if st.session_state.langchain_qa_chain:
                            result = st.session_state.langchain_qa_chain({"query": search_query})
                            answer = result["result"]
                            st.markdown("### AI Answer:")
                            st.info(answer)
                
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    st.error("Please try resetting the application if this problem persists.")
