import os
import streamlit as st
import json
import re
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent, Tool

# Set API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

# Set environment variables
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY

# Constants
MODEL_NAME = "llama3-8b-8192"  # Free Groq model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 2000
TEMPERATURE = 0.1

def validate_requirements_document(text):
    """
    Final validation function for scenario-style requirements
    """
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Structure validation
    structure_checks = [
        r'\bA Customer [a-zA-Z]+\b',  # Scenario pattern
        r'\b(Account|Order|Product|LineItem)\b',  # Key entities
        r'\b(shall be|must be|provides|tracks|reflects)\b'  # System actions
    ]
    
    # Quality attributes
    quality_checks = [
        r'\b(real-time|unaffected|visible|downloadable)\b',
        r'\b\d+ms\b',  # Response times
        r'\b(security|availability)\b'
    ]
    
    structure_score = sum(1 for p in structure_checks if re.search(p, text))
    quality_score = sum(1 for p in quality_checks if re.search(p, text))
    
    if structure_score >= 2 and quality_score >= 1:
        return True, "Valid requirements document"
    
    errors = []
    if structure_score < 2:
        errors.append("Missing required scenario structure elements")
    if quality_score < 1:
        errors.append("Missing quality attributes")
    
    return False, "Invalid document: " + ". ".join(errors)

# -------------- Add this new helper function --------------
def format_bullet_points(text):
    """
    Format bullet points properly for Streamlit markdown.
    This ensures that bullet points (•) are correctly converted to markdown bullet points.
    """
    # Split the text by lines
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Check if the line contains a bullet point
        if "• " in line:
            # Extract the position of the bullet point
            bullet_pos = line.find("• ")
            
            # Calculate leading spaces/indentation
            leading_spaces = line[:bullet_pos].count(' ')
            
            # Replace the bullet with markdown bullet and preserve indentation
            formatted_line = " " * leading_spaces + "* " + line[bullet_pos + 2:]
            formatted_lines.append(formatted_line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)
# ----------------------------------------------------------

# Initialize LLM function
def initialize_llm():
    """Initialize the LLM with appropriate parameters"""
    try:
        return ChatGroq(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=GROQ_API_KEY
        )
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {str(e)}")
        st.error("Check your Groq API key and internet connection")
        return None

def load_document(file_path):
    """Load PDF, DOCX, or TXT files and return documents"""
    try:
        if file_path.endswith('.pdf'):
            docs = PyPDFLoader(file_path).load()
        elif file_path.endswith('.docx'):
            docs = Docx2txtLoader(file_path).load()
        else:
            docs = TextLoader(file_path).load()
            
        return docs
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []

def initialize_vector_store(docs):
    """
    Create vector store from documents for retrieval
    Note: This function is kept for compatibility but is no longer used 
    due to persistent vector store errors
    """
    # We're keeping this function stub for compatibility with existing code
    # but we won't be using vector stores anymore due to the errors
    st.warning("Vector store functionality has been disabled due to compatibility issues.")
    return None

def chunk_document(text, max_chunk_size=4000):
    """Split document into manageable chunks for the LLM"""
    if len(text) <= max_chunk_size:
        return [text]
    
    # Simple splitting by paragraphs to keep context
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

# Define custom tools with proper type annotations
class RequirementsPrioritizer(BaseTool):
    name: str = "requirements_prioritizer"
    description: str = "Prioritizes requirements based on importance and dependency"
    
    def _run(self, text: str) -> str:
        llm = initialize_llm()
        
        prioritization_template = """
        Given these requirements:
        {text}
        
        Prioritize them based on:
        1. Business value
        2. Implementation complexity
        3. Dependencies between requirements
        
        Format as a bullet list with priorities (High, Medium, Low) and brief justification.
        Use proper bullet points (•) for each requirement.
        """
        
        prioritization_prompt = PromptTemplate(
            template=prioritization_template,
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=llm, prompt=prioritization_prompt)
        result = chain.run(text=text)
        
        return result
        
    async def _arun(self, text: str) -> str:
        """Async implementation of run"""
        # For simplicity, using the sync version
        return self._run(text)

class UseCaseDiagramGenerator(BaseTool):
    name: str = "use_case_diagram_generator" 
    description: str = "Generates a use case diagram from requirements"
    
    def _run(self, text: str) -> str:
        llm = initialize_llm()
        
        diagram_template = """
        Given these requirements:
        {text}
        
        Extract actors and use cases, then create a Mermaid diagram showing relationships.
        Format your response as valid Mermaid diagram code with actors, use cases and their relationships.
        
        Example format:
        mermaid
        graph TD
          A[Actor1] --> UC1[Use Case 1]
          A[Actor1] --> UC2[Use Case 2]
          B[Actor2] --> UC3[Use Case 3]
        
        
        """
        
        diagram_prompt = PromptTemplate(
            template=diagram_template,
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=llm, prompt=diagram_prompt)
        result = chain.run(text=text)
        
        # Extract the Mermaid diagram code from the response
        mermaid_match = re.search(r'mermaid\s*(.*?)\s*', result, re.DOTALL)
        if mermaid_match:
            mermaid_code = mermaid_match.group(1)
        else:
            mermaid_code = result
            
        return f"mermaid\n{mermaid_code}\n"
        
    async def _arun(self, text: str) -> str:
        """Async implementation of run"""
        # For simplicity, using the sync version
        return self._run(text)

class AmbiguityDetector(BaseTool):
    name: str = "ambiguity_detector"
    description: str = "Detects ambiguities in requirements"
    
    def _run(self, text: str) -> str:
        llm = initialize_llm()
        
        ambiguity_template = """
        Analyze these requirements for ambiguities:
        {text}
        
        Identify:
        1. Vague terms (e.g., "fast", "user-friendly")
        2. Unclear scope
        3. Conflicting requirements
        4. Missing information
        
        List each ambiguity with the specific text and a brief explanation of why it's ambiguous.
        Use bullet points (•) for each ambiguity found.
        """
        
        ambiguity_prompt = PromptTemplate(
            template=ambiguity_template,
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=llm, prompt=ambiguity_prompt)
        result = chain.run(text=text)
        
        return result
        
    async def _arun(self, text: str) -> str:
        """Async implementation of run"""
        # For simplicity, using the sync version
        return self._run(text)

def create_requirements_agent():
    """Create an agent that can analyze requirements using tools"""
    llm = initialize_llm()
    
    # Define tools
    tools = [
        RequirementsPrioritizer(),
        UseCaseDiagramGenerator(),
        AmbiguityDetector()
    ]
    
    # Define the agent prompt
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert software requirements engineer. 
        You have access to tools that can help analyze requirements.
        Think step by step to determine what information is needed and which tools to use.
        Always format your final response in a clear, well-structured way with headings and sections.
        Use bullet points for lists rather than paragraphs.
        """),
        ("human", "{input}"),
        ("agent", "{agent_scratchpad}")
    ])
    
    # Create the agent
    agent = create_react_agent(llm, tools, agent_prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent_executor

# Simplified direct analysis function without multi-agent approach
def analyze_requirements_directly(text):
    """Analyze requirements directly with a single prompt to avoid multi-agent complications"""
    llm = initialize_llm()
    
    comprehensive_template = """You are an expert software requirements analyst.
    
    Analyze this software requirements document carefully:
    {text}
    
    Provide a comprehensive analysis including:
    1. FUNCTIONAL REQUIREMENTS:
       - List each unique functional requirement with bullet points
    
    2. NON-FUNCTIONAL REQUIREMENTS:
       - List each unique non-functional requirement with bullet points (performance, security, usability, etc.)
    
    3. ACTORS:
       - List each unique actor (user or external system) with bullet points
    
    4. USE CASES:
       - List each unique use case with the associated actor using bullet points
    
    5. ACTOR-USE CASE RELATIONSHIPS:
       - For each actor, list the use cases they interact with using bullet points
    
    6. PRIORITY ANALYSIS:
       - Assign a priority (High, Medium, Low) to key requirements using bullet points
       - Include brief justification based on business value and dependencies
    
    7. AMBIGUITIES:
       - Identify any ambiguous requirements that need clarification using bullet points
    
    8. USE CASE DIAGRAM:
       - Provide a Mermaid diagram code showing actors and their relationships to use cases
    
    Format your response with proper bullet points for each item in the lists, not paragraphs or numbered lists.
    Each bullet point should start with a • character.
    """
    
    comprehensive_prompt = PromptTemplate(
        template=comprehensive_template,
        input_variables=["text"]
    )
    
    chain = LLMChain(llm=llm, prompt=comprehensive_prompt)
    result = chain.run(text=text)
    
    # Post-process to ensure mermaid code is properly formatted
    result = format_mermaid_in_response(result)
    
    return result

def format_mermaid_in_response(text):
    """Ensure Mermaid diagrams are properly formatted with markdown code blocks"""
    # Find Mermaid sections and format them properly
    mermaid_pattern = r'(graph\s+TD|graph\s+LR|sequenceDiagram|classDiagram|erDiagram|gantt|pie|flowchart\s+[TBLR])'
    
    # If we find what looks like mermaid code not in a code block
    matches = re.finditer(mermaid_pattern, text)
    
    # Keep track of positions where we've made replacements
    processed = []
    result = text
    
    for match in matches:
        # Only process if not already in a processed region
        if not any(start <= match.start() <= end for start, end in processed):
            # Find the start of the mermaid code
            pos = match.start()
            
            # Look backwards for a code block start
            code_block_before = text[:pos].rfind("mermaid")
            if code_block_before == -1 or "" in text[code_block_before:pos]:
                # No code block started or a code block was closed, so we need to add one
                
                # Find where the mermaid code might end
                possible_end = text[pos:].find("")
                if possible_end == -1:
                    # No end code block, so find a reasonable ending point
                    end_patterns = ["\n\n", "\r\n\r\n", "\n## ", "\n# "]
                    end_positions = [text[pos:].find(pat) for pat in end_patterns]
                    end_positions = [p for p in end_positions if p != -1]
                    
                    if end_positions:
                        end_pos = pos + min(end_positions)
                    else:
                        end_pos = len(text)
                else:
                    end_pos = pos + possible_end
                
                # Extract the mermaid code
                mermaid_code = text[pos:end_pos]
                
                # Replace with properly formatted code
                formatted_code = f"mermaid\n{mermaid_code}\n"
                result = result[:pos] + formatted_code + result[end_pos:]
                
                # Record this region as processed
                processed.append((pos, pos + len(formatted_code)))
    
    return result

def create_unified_qa_chain(vectorstore):
    """Create a unified chain for document analysis and Q&A"""
    if vectorstore is None:
        return None
    
    llm = initialize_llm()
    
    # Create a prompt that can handle both analysis and questions
    unified_prompt = ChatPromptTemplate.from_template("""
    You are an AI requirements analyst specialized in software engineering.
    
    Use the following pieces of context to answer the question or perform the analysis:
    
    {context}
    
    The user wants: {question}
    
    If the request is for information about the requirements, answer directly.
    If the request is for analysis, provide a thoughtful step-by-step analysis.
    If the request is to generate a diagram or visualization, describe how it would look.
    If the request is to evaluate something, provide a reasoned evaluation with pros and cons.
    
    Format your response with bullet points for lists rather than paragraphs.
    """
    )
    
    # Create a retrieval chain
    document_chain = create_stuff_documents_chain(llm, unified_prompt)
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # Retrieve 4 most relevant chunks
    )
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def ask_question_directly(text, question):
    """Ask a question about requirements using the LLM directly without vectors"""
    llm = initialize_llm()
    
    qa_template = """
    You are an AI requirements analyst specialized in software engineering.
    
    Here are the requirements:
    
    {text}
    
    Answer this question about the requirements: {question}
    
    Provide a clear and direct answer based on the information in the requirements.
    Be thorough but avoid speculation beyond what's in the document.
    Format your response with bullet points (•) for lists rather than continuous paragraphs.
    
    If you're asked to count or identify entities (like actors, use cases, etc.), list each one
    on a separate bullet point line, and provide the total count at the beginning of your answer.
    """
    
    qa_prompt = PromptTemplate(
        template=qa_template,
        input_variables=["text", "question"]
    )
    
    try:
        chain = LLMChain(llm=llm, prompt=qa_prompt)
        result = chain.run(text=text, question=question)
        return result
    except Exception as e:
        st.error(f"Error during direct question answering: {str(e)}")
        # Ultimate fallback - direct call to Groq with minimal context
        try:
            minimal_prompt = f"Based on these requirements: [brief excerpt of requirements] \n\nQuestion: {question}"
            return llm.invoke(minimal_prompt).content
        except Exception as fallback_error:
            return f"Error processing question: {str(fallback_error)}. Please try again with a simpler question."

def generate_use_case_diagram(requirements_text):
    """Generate a Mermaid use case diagram directly"""
    llm = initialize_llm()
    
    diagram_template = """
    Extract actors and use cases from these requirements, then create a Mermaid diagram:
    
    {text}
    
    Your response should be a valid Mermaid diagram code showing actors and their relationship to use cases.
    Use this format:
    mermaid
    graph TD
      classDef actor fill:#f9f,stroke:#333,stroke-width:2px
      classDef usecase fill:#bbf,stroke:#333,stroke-width:1px
      A1[Actor1]:::actor --> UC1(Use Case 1):::usecase
      A1[Actor1]:::actor --> UC2(Use Case 2):::usecase
      A2[Actor2]:::actor --> UC3(Use Case 3):::usecase
    
    
    Include all identified actors and use cases in the diagram.
    """
    
    diagram_prompt = PromptTemplate(
        template=diagram_template,
        input_variables=["text"]
    )
    
    chain = LLMChain(llm=llm, prompt=diagram_prompt)
    result = chain.run(text=requirements_text)
    
    # Extract just the mermaid code
    mermaid_match = re.search(r'mermaid\s*(.*?)\s*', result, re.DOTALL)
    if mermaid_match:
        return f"mermaid\n{mermaid_match.group(1)}\n```"
    else:
        return result

def analyze_text_with_tool(text, tool_type="ambiguity"):
    """Analyze text using a specific tool directly (without agent framework)"""
    llm = initialize_llm()
    
    if tool_type == "ambiguity":
        tool = AmbiguityDetector()
        return tool._run(text)
    elif tool_type == "prioritize":
        tool = RequirementsPrioritizer()
        return tool._run(text)
    elif tool_type == "diagram":
        tool = UseCaseDiagramGenerator()
        return tool._run(text)
    else:
        # Generic analysis if tool type not specified
        template = """
        Analyze these requirements:
        {text}
        
        Provide a comprehensive analysis including:
        1. Key functional requirements
        2. Key non-functional requirements
        3. Main actors and their use cases
        4. Any ambiguities or issues
        
        Format your response with clear headings and bullet points (•).
        Do not use paragraphs for lists.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["text"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(text=text)
        
        return result

def ask_question_about_requirements(text, question):
    """
    Ask a question about the requirements using direct questioning 
    without relying on vector stores which are causing errors
    """
    # Skip the vector store approach completely and go directly to LLM
    return ask_question_directly(text, question)

# Set page configuration
st.set_page_config(
    page_title="AI Requirements Engineer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
def local_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Card-like styling for sections */
        .stApp {
            font-family: 'Roboto', sans-serif;
        }
        
        /* Headers */
        h1 {
            color: #1a237e;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        h2 {
            color: #283593;
            font-weight: 600;
        }
        
        h3 {
            color: #303f9f;
            font-weight: 500;
        }
        
        /* Analysis box styling */
        .analysis-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            border-radius: 10px;
        }
        
        .stTextArea > div > div > textarea {
            border-radius: 10px;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 20px;
            background-color: #3f51b5;
            color: white;
            font-weight: 500;
            padding: 0.5rem 2rem;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #283593;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }
        
        /* File uploader */
        .stFileUploader > div {
            border-radius: 10px;
            padding: 1rem;
            background-color: #e8eaf6;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #e8eaf6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #c5cae9;
            font-weight: bold;
        }
        
        /* Toggle styling */
        .stRadio [data-testid="stRadio"] > div {
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #3f51b5;
        }
        
        /* Card style containers */
        .card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #e8eaf6;
        }
        
        /* Logo and banner */
        .logo-banner {
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Button groups */
        .button-group {
            display: flex;
            gap: 10px;
            margin: 1rem 0;
        }
        
        /* Feature cards */
        .feature-card {
            background-color: #e8eaf6;
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid #3f51b5;
        }
        
        /* Status indicators */
        .status-indicator-success {
            padding: 5px 10px;
            background-color: #c8e6c9;
            color: #2e7d32;
            border-radius: 15px;
            font-weight: 500;
            display: inline-block;
            font-size: 0.8rem;
        }
        
        .status-indicator-warning {
            padding: 5px 10px;
            background-color: #fff9c4;
            color: #f57f17;
            border-radius: 15px;
            font-weight: 500;
            display: inline-block;
            font-size: 0.8rem;
        }
        
        .status-indicator-error {
            padding: 5px 10px;
            background-color: #ffcdd2;
            color: #c62828;
            border-radius: 15px;
            font-weight: 500;
            display: inline-block;
            font-size: 0.8rem;
        }
        
        /* Tabs container */
        .tabs-container {
            margin-top: 1.5rem;
        }
        
        /* Centered container */
        .centered-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem 0;
        }
        
        /* Header container */
        .header-container {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(90deg, #3f51b5 0%, #5c6bc0 100%);
            color: white;
            border-radius: 10px;
        }
        
        /* Header text in white */
        .header-container h1, .header-container h2, .header-container h3 {
            color: white;
        }
        
        /* Step indicator */
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin: 1.5rem 0;
            position: relative;
        }
        
        .step-indicator::before {
            content: "";
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 2px;
            background-color: #e0e0e0;
            z-index: 1;
        }
        
        .step {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            position: relative;
            z-index: 2;
            color: #5c6bc0;
        }
        
        .step.active {
            background-color: #3f51b5;
            color: white;
        }
        
        /* Result box */
        .result-box {
            border-left: 5px solid #3f51b5;
            padding-left: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Highlighted text */
        .highlight {
            background-color: #e8eaf6;
            padding: 2px 5px;
            border-radius: 3px;
        }
        
        /* Table styling */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        
        th, td {
            border: 1px solid #e0e0e0;
            padding: 0.5rem;
            text-align: left;
        }
        
        th {
            background-color: #e8eaf6;
            color: #283593;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    # Apply CSS
    local_css()
    
    # Session state initialization
    session_defaults = {
        'doc_text': "",
        'analysis_results': None,
        'processed_results': None,
        'qa_results': None,
        'last_action': None,
        'prev_input_option': None,
        'input_method': "Enter text"
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Header Section
    st.markdown("""
    <div class="header-container">
        <h1>AI Requirements Engineer Assistant</h1>
        <p>Upload your software requirements document and get AI-powered analysis and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout columns
    col1, col2 = st.columns([1, 2], gap="large")
    
    # ======================
    # Column 1: Input Section
    # ======================
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Input Requirements")
        
        # Track input method changes
        prev_method = st.session_state.input_method
        input_option = st.radio(
            "Choose input method:",
            ["Enter text", "Upload file"],
            key="input_method"
        )
        
        # Reset document state when switching methods
        if st.session_state.input_method != prev_method:
            st.session_state.doc_text = ""
            st.session_state.processed_results = None
            st.session_state.qa_results = None
            st.session_state.last_action = None
        
        # Text input handling
        if input_option == "Enter text":
            requirements_text = st.text_area(
                "Paste your requirements here:",
                height=400,
                value=st.session_state.doc_text,
                key="text_input"
            )
            if requirements_text:
                st.session_state.doc_text = requirements_text
                
        # File upload handling        
        else:
            uploaded_file = st.file_uploader(
                "Upload requirements document (PDF, DOCX, TXT):",
                type=["pdf", "docx", "txt"]
            )
            
            if uploaded_file:
                with st.spinner("Processing document..."):
                    try:
                        temp_file_path = f"temp_{uploaded_file.name}"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        docs = load_document(temp_file_path)
                        if docs:
                            full_text = " ".join([doc.page_content for doc in docs])
                            st.session_state.doc_text = full_text
                            st.success(f"Document loaded: {uploaded_file.name}")
                            
                            with st.expander("Preview document content"):
                                st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)
                                
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")

        # ======================
        # Analysis Controls
        # ======================
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analysis Options")
        
        analysis_type = st.selectbox(
            "Select analysis type:",
            [
                "Comprehensive Analysis",
                "Ambiguity Detection",
                "Requirements Prioritization"            ]
        )
        
        # Analysis execution button
        if st.button("Analyze Requirements", use_container_width=True):
            if st.session_state.doc_text:
                try:
                    # Clear previous results
                    st.session_state.qa_results = None
                    st.session_state.last_action = 'analysis'
                    
                    # Validate document
                    is_valid, validation_msg = validate_requirements_document(st.session_state.doc_text)
                    if not is_valid:
                        st.error(f"Validation Error: {validation_msg}")
                        return
                    
                    # Perform analysis
                    with st.spinner("Analyzing requirements..."):
                        analysis_functions = {
                            "Comprehensive Analysis": analyze_requirements_directly,
                            "Ambiguity Detection": lambda x: analyze_text_with_tool(x, "ambiguity"),
                            "Requirements Prioritization": lambda x: analyze_text_with_tool(x, "prioritize"),
                            "Generate Use Case Diagram": lambda x: analyze_text_with_tool(x, "diagram")
                        }
                        
                        result = analysis_functions[analysis_type](st.session_state.doc_text)
                        
                        # Special handling for Mermaid diagrams
                        if analysis_type == "Generate Use Case Diagram":
                            st.session_state.processed_results = f"```mermaid\n{result}\n```"
                        else:
                            st.session_state.processed_results = format_bullet_points(result)
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
            else:
                st.error("No document loaded for analysis")

        # ======================
        # Q&A Section
        # ======================
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ❓ Requirements Inquiry")
        
        user_question = st.text_input(
            "Ask focused questions about:",
            placeholder="e.g. 'List all actors and their use cases'",
            help="Ask specific questions about requirements elements",
            key="question_input"
        )
        
        # Handle question submission (button or enter key)
        if st.button("Analyze Question", key="qa_button") or st.session_state.question_input:
            if user_question:
                # Clear previous results
                st.session_state.processed_results = None
                st.session_state.last_action = 'qa'
                
                if st.session_state.doc_text:
                    try:
                        # Validate document
                        is_valid, valid_msg = validate_requirements_document(st.session_state.doc_text)
                        if not is_valid:
                            st.error(f"Validation Error: {valid_msg}")
                            return
                        
                        # Process question
                        with st.spinner("Processing Question..."):
                            answer = ask_question_directly(st.session_state.doc_text, user_question)
                            st.session_state.qa_results = format_bullet_points(answer)
                    except Exception as e:
                        st.error(f"Analysis Error: {str(e)}")
                else:
                    st.error("No document loaded for analysis")
            else:
                st.error("Please enter a question")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close feature-card

    # ======================
    # Column 2: Results Display
    # ======================
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Analysis Results")
        
        # Dynamic results display
        if st.session_state.last_action == 'analysis' and st.session_state.processed_results:
            if "```mermaid" in st.session_state.processed_results:
                st.markdown(st.session_state.processed_results, unsafe_allow_html=True)
            else:
                st.markdown(st.session_state.processed_results)
        elif st.session_state.last_action == 'qa' and st.session_state.qa_results:
            st.markdown(st.session_state.qa_results)
        else:
            st.info("""
            **Welcome to the AI Requirements Engineer Assistant!**
            
            1. Upload a document or enter text requirements
            2. Choose an analysis type or ask questions
            3. View results in this panel
            """)
            
        st.markdown('</div>', unsafe_allow_html=True)  # Close card

    # ======================
    # Footer
    # ======================
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; font-size: 0.8rem; color: #666;">
        <p>AI Requirements Engineer Assistant - Powered by Groq LLM API</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
