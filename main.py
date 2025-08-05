import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")


from google import genai
from google.genai import types

# Create an instance of the client 
client = genai.Client(api_key=api_key)

def main():
    if len(sys.argv) < 2:
        print("Invalid input!!!")
        sys.exit(1)
    messages = [
        types.Content(role="user", parts=[types.Part(text=sys.argv[1])]),
    ]
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=messages
    )
    if len(sys.argv) == 3 and sys.argv[2] == "--verbose":
        print(f"User prompt: {sys.argv[1]}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print(response.text)


if __name__ == "__main__":
    main()
