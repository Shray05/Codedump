import os
import subprocess
from dotenv import load_dotenv

# Load environment variables from your .env file (if you renamed it to .env)
load_dotenv('.env')

# Retrieve the API key securely from your environment variables
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables. Please add it to your .env file.")

# Import the Google GenAI client library
from google import genai

def get_ai_response(prompt: str) -> str:
    """
    Uses the Google GenAI Client to generate content based on the given prompt.
    """
    client = genai.Client(api_key=api_key)
    
    # Choose the appropriate model (for example, "gemini-2.0-flash")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    return response.text

def append_response_to_file(filename: str, response_text: str):
    """
    Appends the AI response as a comment to the specified Python file.
    """
    with open(filename, "a", encoding="utf-8") as file:
        file.write("\n# AI Response:\n")
        file.write(response_text)
        file.write("\n")

def commit_changes(filename: str, commit_message: str):
    """
    Stages the file changes, commits them, and pushes to the remote GitHub repository.
    """
    subprocess.run(["git", "add", filename], check=True)
    subprocess.run(["git", "commit", "-m", commit_message], check=True)
    subprocess.run(["git", "push"], check=True)

if __name__ == "__main__":
    # Define your prompt string here.
    prompt = " DO NOT MAKE IT A CODE BLOCK. DO NOT MAKE IT A CODE BLOCK. ONLY MAKE IT PLAIN TEXT. DO NOT MAKE IT A CODE BLOCK. DO NOT MAKE IT A CODE BLOCK. generate a large python function but return as plain text ONLY MAKE IT PLAIN TEXT. DO NOT MAKE IT A CODE BLOCK. DO NOT MAKE IT A CODE BLOCK. ONLY MAKE IT PLAIN TEXT. DO NOT MAKE IT A CODE BLOCK. DO NOT MAKE IT A CODE BLOCK. ONLY MAKE IT PLAIN TEXT."
    
    # Get the AI response to the prompt using the client library.
    ai_response = get_ai_response(prompt)
    print("AI Response obtained:")
    print(ai_response)
    
    # Specify the file to which you want to append the AI response.
    file_to_update = "auto_generated_code.py"
    
    # If the file does not exist, create it with an initial comment.
    if not os.path.exists(file_to_update):
        with open(file_to_update, "w", encoding="utf-8") as file:
            file.write("# This file will be updated with AI responses.\n")
    
    # Append the AI response to the file.
    append_response_to_file(file_to_update, ai_response)
    print(f"Appended AI response to {file_to_update}")
    
    # Automatically commit and push the updated file to GitHub.
    commit_message = "Automatically add AI-generated response to file."
    try:
        commit_changes(file_to_update, commit_message)
        print("Changes committed and pushed to GitHub successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during git commit/push: {e}")
