# News Research Tool 📈

A powerful AI-powered news research application that allows you to analyze multiple news articles from URLs and ask questions about their content. Built with Streamlit, LangChain, and FAISS vector database, this tool uses advanced natural language processing to extract insights from news articles.

## 🎯 Project Overview

The News Research Tool is an intelligent web application that:
- Extracts and processes content from multiple news article URLs
- Creates vector embeddings for semantic search
- Enables natural language Q&A about the articles
- Provides source citations for answers

This tool is perfect for researchers, journalists, students, and anyone who needs to quickly understand and query information from multiple news sources.

## ✨ Features

- **Multi-URL Processing**: Process up to 3 news article URLs simultaneously
- **Intelligent Text Splitting**: Automatically chunks articles for optimal processing
- **Vector Embeddings**: Uses HuggingFace embeddings (all-MiniLM-L6-v2) for semantic understanding
- **FAISS Vector Store**: Fast and efficient similarity search using Facebook AI Similarity Search
- **AI-Powered Q&A**: Ask questions in natural language using Llama 3.3 70B model via Groq
- **Source Citation**: Get answers with references to the original sources
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface
- **Persistent Storage**: Saves vector indices locally for quick reloading

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.13+**: Programming language
- **Streamlit**: Web application framework
- **LangChain**: Framework for building LLM applications
- **FAISS**: Vector similarity search library (Facebook AI)

### AI/ML Components
- **Groq API**: Fast inference for Llama 3.3 70B model
- **HuggingFace Embeddings**: all-MiniLM-L6-v2 for text embeddings
- **LangChain Community**: Document loaders and vector stores

### Key Libraries
- `langchain-groq`: Groq LLM integration
- `langchain-huggingface`: HuggingFace embeddings
- `langchain-community`: Community integrations
- `unstructured`: Document parsing
- `faiss-cpu`: CPU-optimized FAISS

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Python 3.13 or higher** installed on your system
2. **Groq API Key** - Get your free API key from [Groq Console](https://console.groq.com/)
3. **Internet connection** for downloading models and accessing news URLs

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd NewsResearchTool
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The installation may take a few minutes as it downloads ML models and dependencies.

### Step 4: Configure API Key

1. Create a file named `secret_key.py` in the project root directory
2. Add your Groq API key:

```python
GROQ_API_KEY = "your-groq-api-key-here"
```

**Important**: 
- Never commit `secret_key.py` to version control (it's already in `.gitignore`)
- Keep your API key secure and private

### Step 5: Verify Installation

Check that all dependencies are installed correctly:

```bash
python -c "import streamlit; import langchain; print('Installation successful!')"
```

## 🎮 How to Use

### Starting the Application

1. **Activate your virtual environment** (if using one)

2. **Run the Streamlit app**:
```bash
streamlit run main.py
```

3. **Open your browser** - Streamlit will automatically open the app at `http://localhost:8501`

### Using the Tool

#### Step 1: Process URLs
1. In the sidebar, enter up to 3 news article URLs (one per field)
2. Click the **"Process URLs"** button
3. Wait for the processing to complete:
   - Data Loading: Extracts content from URLs
   - Text Splitting: Chunks the articles for processing
   - Embedding Generation: Creates vector embeddings
   - Vector Index Building: Saves to FAISS index

4. You'll see **"Embeddings created successfully!"** when done

#### Step 2: Ask Questions
1. Enter your question in the **"Question"** field
2. Click **"Ask"** or press Enter
3. View the answer along with source citations

### Example Questions

- "What are the main points discussed in these articles?"
- "What is the author's opinion on the topic?"
- "Summarize the key findings from all articles"
- "What are the dates mentioned in the articles?"
- "What are the different perspectives presented?"

## 📁 Project Structure

```
NewsResearchTool/
│
├── main.py                 # Main application file
├── secret_key.py          # API key configuration (not in git)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore            # Git ignore rules
│
├── my_faiss_index/        # Generated vector index (created after first run)
│   ├── index.faiss       # FAISS vector index file
│   └── index.pkl         # Metadata pickle file
│
└── __pycache__/          # Python cache (ignored)
```

## 🔧 Configuration

### Changing the LLM Model

In `main.py`, you can modify the Groq model:

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Change this
    temperature=0.6                    # Adjust creativity (0-1)
)
```

### Adjusting Text Chunking

Modify the text splitter parameters in `main.py`:

```python
recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' '],
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200,      # Overlap between chunks
    length_function=len,
)
```

### Changing Embedding Model

Update the embedding model:

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **FAISS Index Error**
```
Error: could not open .../index.faiss for reading
```
**Solution**: Make sure you've processed URLs first. The index is created after clicking "Process URLs".

#### 2. **API Key Error**
```
Error: Invalid API key
```
**Solution**: 
- Verify your `secret_key.py` file exists
- Check that `GROQ_API_KEY` is correctly set
- Ensure the API key is valid and has credits

#### 3. **URL Loading Fails**
```
Error loading URL
```
**Solution**:
- Verify URLs are accessible and valid
- Check internet connection
- Some websites may block automated access

#### 4. **Module Not Found**
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: 
```bash
pip install -r requirements.txt
```

#### 5. **Slow Processing**
- First-time embedding generation downloads models (~80MB)
- Large articles take longer to process
- Consider reducing `chunk_size` for faster processing

## 🔒 Security Notes

- **Never commit API keys** to version control
- The `secret_key.py` file is in `.gitignore` for your protection
- Consider using environment variables for production:
  ```python
  import os
  GROQ_API_KEY = os.getenv("GROQ_API_KEY")
  ```

## 📊 How It Works

1. **URL Loading**: `UnstructuredURLLoader` fetches and parses HTML content from URLs
2. **Text Splitting**: Articles are split into manageable chunks with overlap
3. **Embedding Creation**: Each chunk is converted to a vector using HuggingFace embeddings
4. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
5. **Question Processing**: User questions are embedded and matched against article chunks
6. **Answer Generation**: Llama 3.3 model generates answers based on retrieved context
7. **Source Citation**: Original sources are tracked and displayed with answers

## 🚧 Future Enhancements

Potential improvements for the project:

- [ ] Support for more than 3 URLs
- [ ] PDF document support
- [ ] Export answers to PDF/Word
- [ ] Conversation history
- [ ] Multiple embedding model options
- [ ] Advanced filtering and search options
- [ ] Batch processing mode
- [ ] User authentication
- [ ] Cloud deployment options

## 📝 License

This project is open source and available for personal and educational use.

## 🙏 Acknowledgments

- **Groq** for providing fast LLM inference
- **HuggingFace** for embedding models
- **LangChain** for the excellent framework
- **Streamlit** for the web framework
- **FAISS** by Facebook AI Research

## 📧 Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Happy Researching! 📚🔍**

