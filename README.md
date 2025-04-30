# AI Requirements Engineer Assistant

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-00B0B9)](https://github.com/langchain-ai/langchain)

![AI Requirements Engineer Banner](https://raw.githubusercontent.com/yourusername/ai-requirements-engineer/main/assets/banner.png)

## 🚀 Overview

The AI Requirements Engineer Assistant is a powerful Streamlit-based application that leverages LLM technology to analyze, validate, and improve software requirements documents. This tool helps software teams identify ambiguities, prioritize requirements, generate use case diagrams, and answer questions about requirements documents—all through an intuitive user interface.

This tool addresses the critical challenges of requirements engineering by bringing AI-powered analysis to early-stage software development.

## ✨ Features

- **📄 Multi-format Document Support**: Upload PDF, DOCX, or TXT requirements documents
- **🔍 Comprehensive Requirements Analysis**: Extract functional and non-functional requirements
- **❓ Interactive Q&A**: Ask specific questions about your requirements
- **📊 Visualization**: Generate Mermaid use case diagrams
- **⚠️ Ambiguity Detection**: Identify vague or unclear requirements
- **🏆 Priority Analysis**: Get AI-recommended priorities for requirements
- **🔄 Simple User Experience**: Clean, intuitive interface with responsive design

## 🔧 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/ai-requirements-engineer.git

# Navigate to the project directory
cd ai-requirements-engineer

# Create a virtual environment 
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🔑 API Keys Setup

The application requires API keys for Groq and HuggingFace:

1. Create a `.env` file in the root directory
2. Add your API keys:
```
GROQ_API_KEY="your_groq_api_key_here"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_key_here"
```

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Navigate to the provided URL (typically http://localhost:8501) and:

1. **Input Requirements**: Either paste your requirements text or upload a document
2. **Choose Analysis Type**: Select from analysis options like ambiguity detection or comprehensive analysis
3. **View Results**: Get instant AI-powered insights about your requirements
4. **Ask Questions**: Interact with your requirements through natural language questions

## 🖥️ Application Interface

The application provides a clean, intuitive interface:

- **Left Panel**: Input methods (text or file upload) and analysis controls
- **Right Panel**: Analysis results and visualization display
- **Interactive Elements**: Question answering, analysis options, and result formatting

## 💻 Technical Architecture

The AI Requirements Engineer Assistant is built on a modern stack:

- **Frontend**: Streamlit for the web interface
- **LLM Integration**: LangChain framework for AI model integration
- **Models**: Groq's Llama3-8b-8192 model for analysis
- **Embedding**: HuggingFace embeddings for text processing
- **Document Processing**: PyPDF, Docx2txt, and custom text processors
- **Visualization**: Mermaid diagrams for use case visualization

## 🔄 Workflow

1. Document parsing and preprocessing
2. Chunking long documents for LLM processing
3. Analysis through specialized prompts and tools
4. Result formatting with proper structure and bullet points
5. Interactive question answering about requirements

## 🚧 Future Enhancements

- **Requirements Versioning**: Track changes between document versions
- **Integration with Issue Trackers**: Connect with JIRA, GitHub Issues, etc.
- **Advanced Diagram Generation**: Class diagrams, sequence diagrams, and ER diagrams
- **Acceptance Criteria Generation**: Auto-generate test scenarios from requirements
- **Multi-Language Support**: Process requirements in different languages
- **Custom Templates**: Define organization-specific requirements templates
- **Collaboration Features**: Comments and multi-user feedback on requirements
- **Export Options**: Generate formal SRS documents in various formats
- **Metrics Dashboard**: Track requirements quality metrics over time

## 🛡️ Data Privacy

All processing happens locally - your requirements documents are not stored or shared.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 🙏 Acknowledgements

- Groq API for providing access to powerful LLMs
- HuggingFace for embeddings and models
- Streamlit for the amazing web framework
- LangChain for simplifying LLM integration


---

Built with ❤️ by the AI Requirements Engineer Team.
