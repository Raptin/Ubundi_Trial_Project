# Ubundi Trial Project - AI Chat Application

A Streamlit-based conversational AI application that allows users to chat with an AI assistant about uploaded documents. The system uses LangChain, OpenAI GPT models, and Pinecone vector database for document processing and intelligent responses.

## 🏗️ System Architecture

This project implements a document-aware conversational AI system with the following components:

- **Frontend**: Streamlit web interface
- **AI Models**: OpenAI GPT-4o-mini for chat responses
- **Vector Database**: Pinecone for document embeddings and retrieval
- **Document Processing**: Support for PDF, TXT, CSV, DOC, DOCX files
- **Conversation Modes**: Multiple AI personality modes (Interview, Storytelling, Fast Facts, Joker, etc.)

## 🛠️ Technology Stack

### Core Dependencies

- **Python 3.11** (via devcontainer)
- **Streamlit** - Web application framework
- **LangChain** - AI application framework
- **OpenAI** - GPT-4o-mini language model
- **Pinecone** - Vector database for embeddings

### Document Processing

- **PyPDF** - PDF file processing
- **Unstructured** - Document parsing
- **Python-docx** - Word document processing
- **Pandas** - Data manipulation

### Reasoning behind these choices

- **Streamlit** - I chose Streamlit because it is a simple and easy to deploy framework. And I did not want to spend too much of my time on the deployment and frontend.
- **LangChain** - I chose LangChain because it seemed like a very versatile framework and had a lot of useful resources online.
- **OpenAI** - I chose OpenAI for my LLM because it feels the most versatile in general conversation. Plus it is the flagship LLM model and will integrate well with other tools.
- **Pinecone** - I chose Pinecone because it seemed user friendly and had a generous free tier available.

## 📁 Project Structure

```
Ubundi_Trial_Project/
├── .streamlit/             # Streamlit configuration
├── embedding/              # Document embedding and processing
├── show_my_work/          # Project documentation/work
├── chatInterface.py       # Main Streamlit application
├── promptBuilder.py       # AI prompt management and response generation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Features

### Chat Interface

- Real-time conversational AI interface
- Multiple AI personality modes
- Document upload and processing
- Chat history persistence

### Document Processing

- Support for multiple file formats (PDF, TXT, CSV, DOC, DOCX)
- Automatic document embedding and indexing
- Context-aware responses based on uploaded documents

### AI Modes

- **Default**: Standard helpful responses
- **Interview Mode**: Job interview simulation
- **Storytelling**: Creative and imaginative responses
- **Fast Facts**: Concise, bullet-pointed information
- **Joker**: Humorous and entertaining responses

## 📝 Usage

1. **Select Chat Mode**: Choose from available AI personality modes
2. **Upload Documents**: Use the sidebar to upload documents for context
3. **Start Chatting**: Ask questions about your documents or have general conversations
4. **View History**: Chat history is maintained throughout the session

### Example questions

What leadership experience do you have?

> I have held several leadership positions throughout my academic and extracurricular activities. Notably, I was the Captain of the second hockey team, where I developed my skills in team management and motivation. Additionally, I served as the Head of House at my residence, managing around 300 students and organizing various sporting and cultural events. This role involved facilitating the welcoming of new Grade 8 students and promoting inclusive interactions among residents.
>
> I was also a member of the school's leadership team, where I contributed to organizing the school's general knowledge competition. My experience as a mentor in a church program further honed my leadership abilities, as I guided and supported youth in their development. These experiences have taught me the importance of leading by example and empowering team members to take ownership of their roles.

What were some of the main findings in your research project?

> In my research project, I focused on evaluating three different models—an end-to-end Neuro-Symbolic (E-E NS) model, a segmented Neuro-Symbolic (SNS) model, and a baseline Convolutional Neural Network (CNN)—on various mathematical operations using MNIST images. Here are some of the main findings:

> Model Performance: The E-E NS model consistently outperformed both the SNS and CNN models across all tested problems, achieving an average accuracy of 95% or greater in addition, subtraction, multiplication, and comparison tasks (greater-than and less-than).

> Statistical Significance: Pairwise Mann-Whitney U tests were conducted to assess the statistical significance of the differences in accuracy between the models. The results indicated that the E-E NS model had a significant advantage over the other models in all problems tested.

> SNS vs. CNN: The SNS model outperformed the CNN model in addition, subtraction, and multiplication tasks. However, the CNN model showed better performance in the binary classification problems (greater-than and less-than), likely due to the simpler nature of these tasks.

> Unexpected Results: Interestingly, both the SNS and CNN models performed better on the multiplication problem compared to addition and subtraction, despite multiplication having a higher number of possible outcomes.

> These findings highlight the effectiveness of the E-E NS model in handling complex mathematical operations and suggest that while the SNS model has its strengths, the choice of model may depend on the specific nature of the task at hand.

What can you tell me about your personality?

> I would describe my personality as curious and outgoing. I have a strong desire to learn and grow, which drives me to seek out new experiences. I enjoy working with people and find fulfillment in helping those around me. My leadership experiences have taught me the importance of building relationships and leading by example, which I believe are essential aspects of my approach to teamwork and collaboration. Overall, I am motivated by making a difference and adding value in both my personal and professional interactions.

## 🕐 What I would improve with more time

### I would have loved to...

- have added more documents to the vector database to make the responses more accurate. However I did not have any on hand and would have had to create/write them from scratch.
- added support for different LLM models.

### I would have liked to...

- give the frontend a bit more styling and personality.
- add a mode where the chatbot acts as an interviewer and performs a mock interview with the user, given a specific job description.

## 🧠 Show my thinking

### Artifacts and Building Blocks

**My Artifacts:**

- **Prompt History**: Extensive conversations with AI agents documented in `show_my_work/cursor_chats.md` and `show_my_work/grok_chats.md`
- **Context Definitions**: Clear specifications for AI personality modes and document processing workflows
- **Agent Instructions**: Detailed prompts for different conversation styles (Interview Mode, Storytelling, Fast Facts, Joker)

### AI vs Manual Implementation Breakdown

**Implementation Breakdown:**

- **AI-Generated**: Core application structure, basic Streamlit components, LangChain integration patterns
- **Manual Refinement**: Error handling, user experience improvements, prompt engineering optimization
- **Hybrid Approach**: AI provided the foundation, I enhanced it with specific business logic and user interface polish

### Experimental Learning Phase

Before implementing the main version, I used several experimental files in the `show_my_work/` folder to learn and iterate:

- **`memory-bot.py`** : Experimented with conversation memory and state management
- **`stateless-bot.py`** : Tested stateless conversation approaches
- **`example_usage.py`** : Prototyped different LangChain patterns and integrations

These experimental files represent my iterative learning process and helped me understand the best approaches before building the production application.
