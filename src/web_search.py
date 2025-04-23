import os
import pdfplumber
import faiss
import numpy as np
import logging
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key! Set it in the .env file or environment variables.")

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ✅ Initialize AI Interview Assistant (using Groq model)
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="AI Interview Assistant specializing in mock interview questions and answers."
)

# ✅ Initialize FAISS for Memory
D = 512  # Embedding dimension
faiss_index = faiss.IndexFlatL2(D)
chat_history = []  # List to store chat history

# ✅ Token Management
TOKEN_LIMIT = 6000  # Example API token limit
RESERVE_FOR_RESPONSE = 500  # Reserve some tokens for AI's response
MAX_EXCHANGES = 3  # Keep last 3 user-AI exchanges


# 📌 Extract Text from Resume
def extract_text_from_pdf(file_path):
    """Extracts text from a resume PDF file."""
    file_path = file_path.strip().strip('"')
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return ""

    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return ""


# 📌 Estimate Token Count
def estimate_tokens(text):
    """Rough estimate of token count based on word count."""
    return len(text.split())


# 📌 Generate Interview Prompts (Enhanced for Conversation)
def generate_prompt(interview_type, resume_text="", job_description="", user_answers=None, **kwargs):
    """
    Generates an adaptive interview prompt based on user inputs and interview context,
    including a formatted conversation history.
    """
    # Base context
    context = "You are an AI interviewer assessing a candidate for a job role."

    # Include candidate details only if provided
    if resume_text:
        context += f"\nCandidate Resume Summary: {resume_text}"
    if job_description:
        context += f"\nJob Description: {job_description}"

    # Format conversation history if available
    if user_answers:
        history_str = ""
        for message in user_answers:
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "")
            history_str += f"{role}: {content}\n"
        context += "\nConversation History:\n" + history_str

    # Get additional parameters
    job_role = kwargs.get('job_role', 'Software Engineer')
    difficulty = kwargs.get('difficulty', 'medium')

    # Build the prompt.
    # The prompt instructs the AI to respond directly to the candidate's last input.
    prompt = (
        f"{context}\n\n"
        f"Based on the above conversation and the interview context for the role of {job_role}, "
        f"please respond to the candidate's latest input. If the candidate is asking for more details "
        f"about the role, provide a detailed explanation. Otherwise, continue the interview by asking a "
        f"relevant {difficulty} difficulty {interview_type} interview question that tests critical skills."
    )

    return prompt


# 📌 AI Chatbot Interaction
def chat_with_agent(user_input, interview_type, resume_text, job_description, job_role, difficulty):
    """
    Interacts with the AI agent while incorporating conversation history into the prompt.
    """
    # Build prompt including the complete conversation history
    prompt_text = generate_prompt(
        interview_type,
        resume_text,
        job_description,
        user_answers=chat_history,
        job_role=job_role,
        difficulty=difficulty
    )

    # Check and trim history if the prompt exceeds the token limit
    while estimate_tokens(prompt_text) > (TOKEN_LIMIT - RESERVE_FOR_RESPONSE):
        if len(chat_history) > 1:
            chat_history.pop(1)  # Remove the oldest non-system message
            prompt_text = generate_prompt(
                interview_type,
                resume_text,
                job_description,
                user_answers=chat_history,
                job_role=job_role,
                difficulty=difficulty
            )
            logging.info(f"Trimming conversation history. New token count: {estimate_tokens(prompt_text)}")
        else:
            break  # No more history to remove

    # Send the prompt to the AI model
    response = agent.run(prompt_text)
    if not response:
        logging.error("AI did not generate a response.")
        return "⚠️ AI could not generate a response."

    ai_response = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    # Store the latest interaction in chat history and FAISS
    store_chat_history(user_input, ai_response)

    return ai_response


# 📌 Store Chat History in FAISS and Local Memory
def store_chat_history(user_input, ai_response):
    """Stores the conversation history and maintains the memory limit."""
    vector = np.random.rand(D).astype('float32')
    faiss_index.add(np.array([vector]))  # Store in FAISS

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": ai_response})

    # Trim history beyond max exchanges (except system messages)
    while len([msg for msg in chat_history if msg["role"] != "system"]) > (MAX_EXCHANGES * 2):
        # Remove the oldest non-system  message
        for i, msg in enumerate(chat_history):
            if msg["role"] != "system":
                chat_history.pop(i)
                break


# 📌 Main Interview Session (Chatbot)
if __name__ == "__main__":
    print("Welcome to AI Interview Chatbot!")

    # User inputs
    job_role = input("Enter job role (e.g., Software Engineer, Data Analyst): ").strip()
    company = input("Enter company name (or 'none' if not applicable): ").strip()

    resume_text = ""
    job_description = ""

    if input("Do you want to upload your resume? (yes/no): ").strip().lower() == "yes":
        file_path = input("Enter the path to your resume (PDF format): ").strip()
        resume_text = extract_text_from_pdf(file_path)

    if input("Do you have a job description to provide? (yes/no): ").strip().lower() == "yes":
        job_description = input("Paste the job description here: ").strip()

    # Build and add a system message with only provided information
    system_content = "You are a job interviewer. Consider the candidate's details in your responses."
    if resume_text:
        system_content += f"\nResume: {resume_text}"
    if job_description:
        system_content += f"\nJob Description: {job_description}"
    system_message = {
        "role": "system",
        "content": system_content
    }
    chat_history.append(system_message)  # Add system prompt to history

    interview_type = input(
        "Enter interview type (e.g., technical, behavioral, or 'n/a' for open discussion): ").strip().lower()
    difficulty = input("Choose difficulty level (easy, medium, hard): ").strip()

    while True:
        user_input = input("\nYour response (type 'exit' to end the interview): ").strip()
        if user_input.lower() == "exit":
            print("Interview session ended. Good luck with your preparation!")
            break

        response = chat_with_agent(user_input, interview_type, resume_text, job_description, job_role, difficulty)
        print("\n🔹 AI Response:\n", response)
