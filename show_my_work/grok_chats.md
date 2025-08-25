# Chats with Grok

---

## Session Information

### Chat number 1

**Model:** Grok-4 (Expert)  
**URL:** https://grok.com/share/bGVnYWN5LWNvcHk%3D_942cf61d-8e61-41dd-aea5-48924b6511dc

---

<div align="right">

**👤 User**

> I want to build a chatbot that can act as an interactive version of myself, that uses Retrieval-Augmented Generation to gather information on me from differing data sources. The chatbot will be used by potential employers to be able to interact and learn a bit more about me during the interview process. Attached is the project specifications. Help me create a step by step gameplan in how to complete this project. And also look at the different tech stacks I can use to complete the project (Pros/cons)

📎 **Attachment:** `Ubundi_Trial_Project_Specifications.txt`

</div>

<div align="left">

**🤖 Assistant**

### Step-by-Step Gameplan to Complete the Ubundi Trial Project

Based on the project specifications in the attached document, here's a structured, time-efficient plan to build your Personal Codex Agent. Aim to allocate your 6-8 hours as follows: ~2 hours on data preparation and planning, ~3-4 hours on implementation, ~1 hour on deployment and testing, and ~1 hour on documentation and artifacts. Focus on core requirements first (RAG-based chatbot with UI), then add one bonus if time allows (e.g., a simple mode switcher). Use AI tools like GitHub Copilot or Claude for code generation to speed up development—document these interactions for the "Show Your Thinking" section.

1. **Understand and Plan (30-45 minutes)**:

   - Review the objective: Build a context-aware agent that answers personal questions using your data, speaking in your voice.
   - Identify key questions from the spec (e.g., "What kind of engineer are you?") and brainstorm 2-3 more based on your background.
   - Gather inputs: Prepare your CV/resume (e.g., as PDF or Markdown). Collect 2-3 supporting documents (e.g., a blog post snippet, code sample with comments, personal notes on work values). Organize them in a folder for easy ingestion—ensure they're authentic and reflective.
   - Decide on RAG: Since it's recommended, use it to retrieve relevant chunks from your documents for accurate, grounded responses. If skipping, note in README: "Future extension: Integrate FAISS for vector storage and retrieval to handle larger datasets dynamically."
   - Sketch system design: Data ingestion → Chunking/Embedding → Vector store → Retrieval → LLM generation → UI for interaction.
   - Choose tech stack (see below for options). Prioritize simplicity: e.g., LangChain for RAG, Streamlit for UI, FAISS for local vector DB, OpenAI/Groq for LLM.
   - Set up a GitHub repo for version control and to host artifacts.

2. **Prepare Data and Ingest (45-60 minutes)**:

   - Convert documents to text if needed (e.g., extract PDF text using PyPDF2).
   - Chunk data: Split into small segments (e.g., 300-500 words) for better retrieval.
   - Embed chunks: Use an embedding model (e.g., OpenAI's text-embedding-ada-002 or Hugging Face's sentence-transformers) to create vectors.
   - Store in vector DB: Index vectors in FAISS or Pinecone for fast similarity search.
   - Test ingestion: Load a sample document and verify embeddings are generated correctly.

3. **Build the RAG Core (1-2 hours)**:

   - Set up retrieval: When a query comes in, retrieve top-k relevant chunks based on cosine similarity.
   - Augment generation: Pass retrieved context to an LLM (e.g., GPT-4o mini) with a prompt like: "Answer as [Your Name], using this context: {context}. Speak authentically and refer to your experiences."
   - Make it agentic: Add basic chaining (e.g., if no relevant data, fallback to a default response) or simple reflection for bonus questions like "Where do I need to grow?"
   - Use AI assistance: Prompt Copilot/Claude with: "Generate Python code for a RAG pipeline using LangChain, FAISS, and OpenAI to query personal docs." Edit and log the prompt/response/code for artifacts.
   - Handle voice: Fine-tune prompts to mimic your style (e.g., casual, reflective) based on your documents.

4. **Develop the UI and Interaction (1 hour)**:

   - Build a simple chatbot interface: Use Streamlit/Replit for a text input box and response display.
   - Integrate RAG: On user input, trigger retrieval and generation, then show the answer.
   - Add optional bonus: Implement a dropdown for modes (e.g., "Interview mode" for concise answers via prompt modifiers). Test with sample questions.
   - Ensure it feels personal: Responses should reference specific document snippets (e.g., "From my CV, I specialized in AI engineering at...").

5. **Deploy the Agent (30-45 minutes)**:

   - Choose a platform: Streamlit Sharing, Vercel (if using Next.js), or Replit for quick hosting.
   - Push code to GitHub, then deploy. Test accessibility and interactions.
   - Handle updates: For bonus, add a file upload feature to extend data easily.

6. **Test and Refine (30 minutes)**:

   - Run sample questions: Verify accuracy, voice, and context usage. Debug hallucinations by strengthening retrieval.
   - Edge cases: Test unrelated queries (e.g., fallback: "I don't have info on that, but based on my values...").
   - Time check: If under budget, add one bonus like self-reflection mode.

7. **Document and Submit (45-60 minutes)**:
   - README: Explain setup (e.g., "pip install langchain streamlit faiss-cpu"), design choices (e.g., "LangChain for modular RAG"), sample Q&A (e.g., Question: "Strongest skills?" Expected: Bullet list from CV), improvements (e.g., "Add multi-modal support for images in docs"), and artifacts.
   - Show Your Thinking: Include prompt histories (e.g., "Prompt to Claude: Build RAG chain → Response: Code snippet → My edits: Added custom prompt"), sub-agents (e.g., "Retrieval agent for doc search"), commit messages (e.g., "AI-generated RAG core, manual tweaks for voice"), and breakdown (e.g., "80% AI-generated via Copilot, 20% manual for personalization").
   - Video walkthrough: Record a 3-5 min screen share explaining the project, demoing the UI, and highlighting artifacts.
   - Submit: GitHub link, deployment URL, video, and any extras.

This plan draws from standard RAG tutorials, emphasizing agentic elements like routing and context building for personalization.

### Tech Stacks for the Project: Pros/Cons

Here are popular stacks for building a RAG-based personal chatbot, focused on Python for simplicity and the project's time constraints. I prioritized options compatible with the suggested deployment platforms (Replit, Streamlit, Vercel). Common components: LLM (OpenAI/Groq), Embeddings (Hugging Face/OpenAI), Vector DB (FAISS for local/free, Pinecone for cloud).

#### 1. **LangChain + FAISS + Streamlit + OpenAI (Deployment: Streamlit Sharing)**

- **Description**: LangChain for RAG orchestration, FAISS for vector storage, Streamlit for UI, OpenAI for generation.
- **Pros**:
  - Highly modular and flexible for chaining retrieval/generation; extensive integrations (e.g., easy document loaders).
  - Rapid prototyping: Build interactive UIs quickly without frontend expertise.
  - Community support: Abundant tutorials for personal RAG apps.
  - Cost-effective: FAISS is local and free; Streamlit deploys in minutes.
- **Cons**:
  - Boilerplate-heavy: Can feel over-engineered for simple projects, leading to steeper learning if new to it.
  - Performance: Streamlit redraws on state changes, which may lag for complex chatbots; not ideal for high-traffic scale.
  - Dependency bloat: Many unused features in LangChain.

#### 2. **LlamaIndex + FAISS + Streamlit + Groq (Deployment: Streamlit Sharing)**

- **Description**: LlamaIndex for focused RAG indexing/retrieval, FAISS for storage, Streamlit UI, Groq for fast/cheap LLM inference.
- **Pros**:
  - Simpler than LangChain for pure RAG: Excels at data ingestion and query engines, making it intuitive for personal docs.
  - Efficient: Better optimized for retrieval from unstructured data like CVs/blogs.
  - Quick setup: Minimal code for indexing; pairs well with Streamlit for beginner-friendly deployment.
  - Scalable basics: Easy to extend for bonuses like data updates.
- **Cons**:
  - Less flexible: Fewer tools for advanced agent chaining compared to LangChain.
  - Integration limits: May require extra work for custom modes or multi-agent setups.
  - Community smaller: Fewer resources than LangChain, though growing.

#### 3. **LangChain + Pinecone + Next.js + OpenAI (Deployment: Vercel)**

- **Description**: LangChain for RAG, Pinecone for cloud vector DB, Next.js for React-based UI, OpenAI for LLM.
- **Pros**:
  - Production-ready: Vercel offers fast, auto-scaling deploys with Git integration; great for AI apps with edge functions.
  - Robust: Pinecone handles large datasets well; Next.js enables responsive, customizable UIs (e.g., chat history).
  - AI-friendly: Seamless with Vercel's AI SDK for streaming responses.
  - Global performance: Edge network reduces latency.
- **Cons**:
  - Steeper curve: Requires frontend knowledge (React); more setup time than Streamlit.
  - Cost: Pinecone and Vercel have usage-based pricing that can add up for queries.
  - Overkill for prototypes: If not scaling, the complexity might exceed the 6-8 hour budget.

#### 4. **Replit's Built-in AI + LlamaIndex + OpenAI (Deployment: Replit)**

- **Description**: Use Replit's IDE with integrated AI (e.g., Replit Agent for code gen), LlamaIndex for RAG, OpenAI for LLM; host directly on Replit.
- **Pros**:
  - All-in-one: Cloud-based IDE with AI assistance for coding; easy collaboration and deployment without local setup.
  - Beginner-friendly: Built-in DB and tools reduce boilerplate; great for quick prototypes.
  - Free tier viable: Supports personal projects with minimal costs.
  - Educational: Aligns with "Show Your Thinking" by logging AI interactions natively.
- **Cons**:
  - Performance limits: Free plans have resource caps; slower for complex RAG with many docs.
  - Dependency on platform: Locked into Replit's ecosystem; harder to migrate.
  - Bugs/unreliability: Some users report issues with AI outputs or stability for larger projects.

#### Recommendations

- **Best for Beginners/Time-Constrained**: LangChain + Streamlit—quick to learn, aligns with project simplicity.
- **If Focusing on RAG Depth**: LlamaIndex + Streamlit—streamlined for document-based agents.
- **For Polish/Deployment**: LangChain + Vercel—if comfortable with JS, for a more professional UI.
- Avoid overcomplicating: Stick to free/open-source where possible to fit the stipend. Test stacks via quick prototypes if unsure.

**Citations:**

- [Practical Guide to RAG Chatbot](https://thenewstack.io/a-practical-guide-to-building-a-rag-powered-chatbot/)
- [Top LLM + RAG Projects 2025](https://ai.plainenglish.io/top-8-llm-rag-projects-for-your-ai-portfolio-2025-c721a5e37b43)
- [LangChain vs LlamaIndex for RAG](https://www.clickittech.com/ai/langchain-vs-llamaindex-for-rag/)
- [Pros/Cons of Streamlit](https://digitaldefynd.com/IQ/pros-cons-of-streamlit/)
- [Vercel for AI Apps](https://www.upsilonit.com/blog/how-to-build-ai-chatbots)
- [Replit Pros/Cons](https://aloa.co/blog/replit)
- [Step-by-Step RAG Guide](https://python.langchain.com/docs/tutorials/rag/)
- [Agentic RAG Tutorial](https://www.datacamp.com/tutorial/agentic-rag-tutorial)

</div>

<div align="right">

**👤 User**

> Someone suggested I use aws bedrock as a potential solution. Do you think this is a good idea? What are the pros and cons?

</div>

<div align="left">

**🤖 Assistant**

Using AWS Bedrock for your Personal Codex Agent project, as outlined in the Ubundi Trial Project specifications, is a viable option, especially given its support for Retrieval-Augmented Generation (RAG) and integration with foundation models (FMs). Below, I evaluate whether AWS Bedrock is a good fit for your project, considering the 6-8 hour time constraint, the need for a simple UI, and the requirement to showcase an AI-native workflow. I also provide a detailed pros and cons analysis of using AWS Bedrock for this RAG-based chatbot, drawing on web sources and the project requirements.

### Is AWS Bedrock a Good Idea for Your Project?

AWS Bedrock is a strong candidate for your project if you prioritize a fully managed service that simplifies RAG implementation and integrates well with a cloud ecosystem. It aligns with the project’s core requirements: building a context-aware chatbot using RAG to answer personal questions based on your CV and supporting documents, deploying a simple UI, and showcasing AI-assisted development. Bedrock’s managed RAG workflow, access to high-performing FMs (e.g., Anthropic Claude), and integration with AWS services like S3 and OpenSearch make it suitable for rapid prototyping within the time budget. However, its complexity, potential costs, and learning curve may pose challenges for a beginner or a short timeline. If you’re comfortable with AWS or have prior cloud experience, Bedrock can streamline development and impress evaluators with a professional, scalable solution. Otherwise, simpler stacks (e.g., LangChain + Streamlit) might be safer for the time constraint.

### Pros and Cons of Using AWS Bedrock for RAG in Chatbot Development

#### Pros

1. **Fully Managed RAG Workflow**:

   - Bedrock’s Knowledge Bases provide an end-to-end RAG solution, handling data ingestion, embedding creation, vector storage, retrieval, and prompt augmentation without custom integrations. This reduces coding effort for document processing (e.g., your CV, blog posts) and aligns with the project’s RAG recommendation.[](https://aws.amazon.com/blogs/machine-learning/build-a-contextual-chatbot-application-using-knowledge-bases-for-amazon-bedrock/)[](https://aws.amazon.com/bedrock/knowledge-bases/)
   - Supports multiple vector stores (e.g., Amazon OpenSearch Serverless, Pinecone), making it flexible for indexing your personal documents.[](https://aws.amazon.com/bedrock/knowledge-bases/)
   - Built-in citation support improves transparency and reduces hallucinations, ensuring responses are grounded in your data (e.g., “Based on my CV, my strongest skill is…”).[](https://aws.amazon.com/blogs/machine-learning/build-a-contextual-chatbot-application-using-knowledge-bases-for-amazon-bedrock/)

2. **Access to High-Performing Foundation Models**:

   - Offers models like Anthropic Claude, Amazon Titan, and Cohere, which are optimized for tasks like text generation and chatbot applications. Claude Instant 1.2, for example, is fast and capable for conversational tasks, fitting your need for a personal, authentic voice.[](https://aws.amazon.com/blogs/machine-learning/build-a-contextual-chatbot-application-using-knowledge-bases-for-amazon-bedrock/)[](https://www.datacamp.com/tutorial/aws-bedrock)
   - Unified API simplifies model selection, allowing you to experiment with different FMs in the AWS console to match your tone (e.g., professional for “Interview mode”).[](https://aws.amazon.com/bedrock/faqs/)

3. **Integration with AWS Ecosystem**:

   - Seamlessly connects with S3 for storing your documents, OpenSearch for vector storage, and Lambda/API Gateway for UI integration (e.g., with Streamlit hosted on AWS). This suits the project’s data inputs (PDFs, text files) and deployment needs.[](https://aws.amazon.com/solutions/guidance/building-custom-chatbots-for-order-recommendations-using-agents-for-amazon-bedrock/)[](https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/develop-a-fully-automated-chat-based-assistant-by-using-amazon-bedrock-agents-and-knowledge-bases.html)
   - Supports serverless architectures, reducing infrastructure management and aligning with the project’s focus on rapid prototyping.[](https://aws.amazon.com/solutions/guidance/building-custom-chatbots-for-order-recommendations-using-agents-for-amazon-bedrock/)[](https://aws.amazon.com/bedrock/)

4. **Security and Compliance**:

   - Bedrock ensures data privacy (no data used for model training, encrypted in transit/at rest), which is ideal for handling personal data like your CV. It supports compliance standards (e.g., GDPR, HIPAA), adding professionalism to your submission.[](https://aws.amazon.com/bedrock/)[](https://aws.amazon.com/bedrock/faqs/)
   - Guardrails block harmful content and enhance factual accuracy, ensuring your chatbot’s responses are safe and reliable for potential employers.[](https://aws.amazon.com/blogs/machine-learning/new-capabilities-make-it-easier-to-use-amazon-bedrock-to-build-and-scale-generative-ai-applications-and-deliver-impact/)[](https://aws.amazon.com/bedrock/pricing/)

5. **Scalability and Future-Proofing**:

   - Bedrock’s serverless nature and integration with tools like Bedrock Agents make it easy to extend for bonus features (e.g., self-reflective mode or data updates) or scale for larger datasets.[](https://thedeveloperspace.com/generative-ai-on-aws/)[](https://aws.amazon.com/bedrock/agents/)
   - Supports multi-modal data (e.g., text, images), allowing you to incorporate code snippets or visual elements from your documents if time permits.[](https://aws.amazon.com/bedrock/knowledge-bases/)

6. **AI-Native Workflow Support**:
   - Bedrock’s playgrounds and console let you experiment with prompts and models, aligning with the “Show Your Thinking” requirement. You can log prompt iterations and model outputs as artifacts to demonstrate AI collaboration.[](https://www.datacamp.com/tutorial/aws-bedrock)[](https://addepto.com/blog/amazon-bedrock-a-users-guide-to-benefits-and-utilization/)

#### Cons

1. **Learning Curve and Setup Complexity**:

   - Bedrock requires familiarity with AWS services (e.g., S3, IAM, Lambda), which can be daunting if you’re new to the platform. Setting up Knowledge Bases and permissions may consume significant time (1-2 hours) within the 6-8 hour budget.[](https://docs.aws.amazon.com/decision-guides/latest/bedrock-or-sagemaker/bedrock-or-sagemaker.html)
   - The AWS console and CDK (Cloud Development Kit) for deployment add overhead compared to simpler tools like Streamlit or Replit, which are more beginner-friendly.[](https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/develop-a-fully-automated-chat-based-assistant-by-using-amazon-bedrock-agents-and-knowledge-bases.html)

2. **Potential Costs**:

   - While Bedrock’s pricing is pay-per-use, costs can add up for API calls (e.g., $0.002 per SQL query generation, $0.008-$0.024 per 1K tokens for Claude). The R1000 stipend (~$55 USD) may cover prototyping, but unoptimized usage (e.g., frequent queries) could exceed it.[](https://aws.amazon.com/bedrock/pricing/)
   - Free-tier limits are restrictive, and enabling model access (e.g., Claude) requires AWS account setup, which takes time.[](https://aws.amazon.com/blogs/machine-learning/build-generative-ai-chatbots-using-prompt-engineering-with-amazon-redshift-and-amazon-bedrock/)

3. **Time Constraints**:

   - Configuring Bedrock’s Knowledge Bases, embeddings (e.g., Amazon Titan), and integrations (e.g., Lambda for UI) may take longer than local solutions like FAISS or LlamaIndex. This risks eating into the time needed for UI development or documentation.[](https://aws.amazon.com/blogs/machine-learning/build-a-contextual-chatbot-application-using-knowledge-bases-for-amazon-bedrock/)[](https://aws.amazon.com/blogs/machine-learning/automate-chatbot-for-document-and-data-retrieval-using-amazon-bedrock-agents-and-knowledge-bases/)
   - Debugging integrations (e.g., S3 to OpenSearch) can be complex, especially under time pressure.

4. **Dependency on AWS Ecosystem**:

   - Bedrock locks you into AWS tools, reducing portability compared to open-source stacks (e.g., LangChain + FAISS). If evaluators prefer platform-agnostic solutions, this could be a drawback.[](https://www.stormit.cloud/blog/amazon-bedrock-vs-chatgpt/)
   - Limited offline capability: Requires internet access for API calls, unlike local setups (e.g., FAISS on Replit).

5. **Limited UI Flexibility**:

   - Bedrock focuses on backend AI; you’ll need additional AWS services (e.g., API Gateway, Lambda) or external tools (e.g., Streamlit) for the UI, adding complexity compared to all-in-one platforms like Replit.[](https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/develop-a-fully-automated-chat-based-assistant-by-using-amazon-bedrock-agents-and-knowledge-bases.html)[](https://medium.com/%40lorevanoudenhove/how-to-build-enterprise-scale-generative-ai-agents-with-aws-bedrock-a-comprehensive-guide-a8b643cd97d4)
   - Building a polished UI (e.g., with mode switching) may require more frontend work, which could be challenging in 6-8 hours.

6. **Overkill for Small-Scale Project**:
   - Bedrock’s enterprise-grade features (e.g., AgentCore, multi-agent collaboration) are more suited for large-scale applications than a personal chatbot with a few documents. Simpler tools like LlamaIndex may be more time-efficient for prototyping.[](https://docs.aws.amazon.com/decision-guides/latest/bedrock-or-sagemaker/bedrock-or-sagemaker.html)[](https://thedeveloperspace.com/generative-ai-on-aws/)

### Recommendations

- **Use Bedrock If**: You’re familiar with AWS, want a robust RAG solution, and aim to impress with a professional, cloud-based deployment. Pair it with Streamlit (hosted on AWS ECS or externally) for a quick UI and use Claude for generation to match your voice. Focus on core RAG functionality and one bonus (e.g., data update via S3 uploads) to stay within time.
- **Alternative If Time/Experience-Constrained**: Opt for LangChain/LlamaIndex + FAISS + Streamlit (deployed on Streamlit Sharing). This avoids AWS setup overhead, is free/local for vector storage, and is easier for beginners. You can note in the README: “Future work: Migrate to AWS Bedrock for scalable RAG with managed Knowledge Bases.”
- **Hybrid Approach**: Use Bedrock for RAG backend (S3 + Knowledge Bases) and Streamlit for UI, deployed separately. This balances Bedrock’s strengths with a simple frontend, but test setup time early.
- **AI Artifacts**: Use Bedrock’s console to experiment with prompts (e.g., “Answer as a candidate with this CV context…”), log these as artifacts, and use GitHub Copilot for Lambda/UI code. Document which parts were AI-generated vs. manually edited.

**Final Answer**:
AWS Bedrock is a good idea if you’re comfortable with AWS and want a scalable, managed RAG solution that showcases cloud expertise. Its strengths include seamless RAG, powerful FMs, and AWS integrations, but the learning curve, costs, and setup time are concerns for a 6-8 hour project. For beginners or to maximize time, a simpler stack (LangChain + Streamlit) may be safer, with Bedrock as a noted future extension.

</div>

---

### Chat number 2

<div align="right">

**👤 User**

> I am building a chatbot that uses RAG to answer questions about a candidate using documents like his CV. I now want to add the option to add aditional context behind the scenes to my chatbot (like adding a joker mode where he responds with jokes). However as soon as I add the extra prompt to the front of my question it loses the ability to "use" the RAG. Why is this? and how can i fix it?

</div>

<div align="left">

**🤖 Assistant**

Adding mode instructions to the query front modifies the retrieval input, causing irrelevant or failed document matches in RAG.

Fix: Retrieve using the clean original query, then prepend mode instructions (e.g., "Respond with jokes") to the system prompt or after context in the final LLM input.

</div>

<div align="right">

**👤 User**

> how would i implement this using langchain?

</div>

<div align="left">

**🤖 Assistant**

To implement RAG with LangChain while adding a mode like "joker mode" without disrupting retrieval:

1. **Separate Query and Mode**: Use the original user query for RAG retrieval, then combine retrieved context with mode instructions (e.g., "Respond with jokes") for the final LLM prompt.

2. **LangChain Implementation**:
   - **Retriever**: Use a LangChain retriever (e.g., `VectorStoreRetriever`) to fetch relevant documents based on the clean user query.
   - **Prompt Template**: Create a `PromptTemplate` that includes the retrieved context, mode instruction, and user query.
   - **Chain**: Build a chain to process retrieval and generation separately.

### Example Code

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize retriever (e.g., FAISS vector store with CV documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("cv_index", embeddings)
retriever = vectorstore.as_retriever()

# Define prompt template
prompt_template = """
{context}

Instructions: {mode_instruction}
Question: {question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "mode_instruction", "question"],
    template=prompt_template
)

# Initialize LLM
llm = OpenAI()

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Function to handle query with mode
def answer_with_mode(query, mode="normal"):
    # Retrieve documents using clean query
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Set mode instruction
    mode_instruction = "Respond with jokes and humor." if mode == "joker" else "Respond professionally."

    # Run chain
    response = chain.run(
        context=context,
        mode_instruction=mode_instruction,
        question=query
    )
    return response

# Example usage
query = "What is the candidate's experience?"
response_normal = answer_with_mode(query, mode="normal")
response_joker = answer_with_mode(query, mode="joker")
print("Normal:", response_normal)
print("Joker:", response_joker)
```

### Key Points

- **Retriever**: Uses clean `query` for document retrieval to ensure RAG works correctly.
- **Mode Instruction**: Added in `mode_instruction` variable, applied only in the prompt, not retrieval.
- **Prompt Template**: Combines context, mode, and query for flexible responses.
- **Chain**: `LLMChain` handles the final generation, keeping retrieval separate.

This ensures RAG retrieves relevant documents while allowing mode-based response styles like "joker mode."

</div>
