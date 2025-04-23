import os
import pdfplumber
import faiss
import numpy as np
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key! Set it in the .env file or environment variables.")

# ✅ Initialize AI Interview Assistant (using Groq model)
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="AI Interview Assistant specializing in mock interview questions and answers."
)

# ✅ Initialize FAISS for Memory
D = 512  # Embedding dimension
faiss_index = faiss.IndexFlatL2(D)
chat_memory = []  # Stores recent chat interactions

# 📌 Extract Text from Resume
def extract_text_from_pdf(file_path):
    """Extracts text from a resume PDF file."""
    file_path = file_path.strip().strip('"')
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return ""

    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return ""


# 📌 Generate Interview Prompts (Optimized)
def generate_prompt(interview_type, resume_text, job_description, user_answers=None, **kwargs):
    """Generates an adaptive interview prompt based on user inputs and interview context."""

    # Initialize context with a base system message
    context = "You are an AI interviewer assessing a candidate for a job role."

    # Only include resume if it was provided
    if resume_text:
        context += f"\nCandidate Resume Summary: {resume_text}"

    # Only include job description if it was provided
    if job_description:
        context += f"\nJob Description: {job_description}"

    # Include previous answers only if they exist
    if user_answers:
        context += f"\nPrevious Answers: {user_answers}"

    # Add interview question context
    job_role = kwargs.get('job_role', 'Software Engineer')
    difficulty = kwargs.get('difficulty', 'medium')

    prompt = f"{context}\n\nYou are now conducting an interview for the role of {job_role}.\n" \
             f"Ask a {difficulty} difficulty interview question based on the candidate’s experience and the job description (if available). " \
             f"Ensure the question tests relevant skills and aligns with industry standards."

    return prompt


# 📌 Generate Interview Prompts (Optimized)
# def generate_prompt(interview_type, resume_text, job_description, **kwargs):
#     """Generates an interview prompt based on user inputs and chat history."""
#
#     # Keep only the last 3 exchanges for context (to avoid hitting token limits)
#     recent_conversation = "\n".join([f"User: {chat['user']}\nAI: {chat['ai']}" for chat in chat_memory[-3:]])
#
#     # Context from resume & job description
#     context = f"Resume Summary: {resume_text}\nJob Description: {job_description}\n" if resume_text else "No resume provided.\n"
#     context += f"Recent Conversation:\n{recent_conversation}\n" if recent_conversation else ""
#
#     # Interview Parameters
#     job_role = kwargs.get('job_role', 'Software Engineer')
#     difficulty = kwargs.get('difficulty', 'medium')
#
#     # Dynamic question generation
#     prompt = f"{context}You are an AI interviewer assessing a candidate for the role of {job_role}.\n" \
#              f"Ask a {difficulty} difficulty interview question based on the user's experience and job description.\n" \
#              f"Ensure the question is specific to the candidate’s background while remaining balanced in complexity."
#
#     return prompt

# 📌 AI Chatbot Interaction
def chat_with_agent(interview_type, resume_text, job_description, **kwargs):
    """AI Chatbot interacts with users, asks & answers interview questions while keeping short-term memory."""

    # Generate the interview prompt
    prompt = generate_prompt(interview_type, resume_text, job_description, **kwargs)

    if not prompt:
        print("⚠️ Error: No prompt generated.")
        return "⚠️ Invalid input or parameters."

    # Send the prompt to the AI model
    response = agent.run(prompt)

    if not response:
        print(f"⚠️ Error: AI did not generate a response.")
        return "⚠️ AI could not generate a response."

    # Extract AI response
    ai_response = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    # Store chat history
    store_chat_history(prompt, ai_response)

    return ai_response

# 📌 Store Chat History in FAISS
def store_chat_history(user_input, ai_response):
    """Stores conversation history efficiently while limiting token size."""
    vector = np.random.rand(D).astype('float32')
    faiss_index.add(np.array([vector]))

    # Keep chat memory limited to the last 10 exchanges to prevent bloating
    chat_memory.append({"user": user_input, "ai": ai_response})
    if len(chat_memory) > 10:
        chat_memory.pop(0)  # Remove the oldest chat to keep memory size in check

# 📌 Main Interview Session (Chatbot)
if __name__ == "__main__":
    print("Welcome to AI Interview Chatbot!")

    job_role = input("Enter job role (e.g., Software Engineer, Data Analyst): ").strip()
    company = input("Enter company name (or 'none' if not applicable): ").strip()

    resume_text = ""
    job_description = ""

    if input("Do you want to upload your resume? (yes/no): ").strip().lower() == "yes":
        file_path = input("Enter the path to your resume (PDF format): ").strip()
        resume_text = extract_text_from_pdf(file_path)

    if input("Do you have a job description to provide? (yes/no): ").strip().lower() == "yes":
        job_description = input("Paste the job description here: ").strip()

    interview_type = input("Enter interview type (e.g., technical, behavioral): ").strip().lower()
    difficulty = input("Choose difficulty level (easy, medium, hard): ").strip()

    while True:
        response = chat_with_agent(interview_type, resume_text, job_description, job_role=job_role, difficulty=difficulty)
        print("\n🔹 AI Response:\n", response)

        user_input = input("\nYour response (type 'exit' to end the interview): ").strip()
        if user_input.lower() == "exit":
            print("Interview session ended. Good luck with your preparation!")
            break

        # Store user response and continue conversation
        store_chat_history(user_input, response)

