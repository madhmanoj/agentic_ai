import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.get_files_info import schema_get_files_info, get_files_info
from functions.get_file_content import schema_get_file_content, get_file_content
from functions.run_python import schema_run_python_file, run_python_file
from functions.write_file import schema_write_file, write_file

# Load environment variables
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

# Create an instance of the client 
client = genai.Client(api_key=api_key)


def call_function(function_call_part: types.FunctionCall, verbose=False):
    if verbose:
        print(f"Calling function: {function_call_part.name}({function_call_part.args})")
    else:
        print(f" - Calling function: {function_call_part.name}")
    
    keyword_arguments = function_call_part.args 
    function_name = function_call_part.name

    keyword_arguments["working_directory"] = "./calculator"

    match function_call_part.name:

        case "get_files_info":
            result = get_files_info(**keyword_arguments)
            return types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=function_name,
                        response={"result": result},
                    )
                ],
            )
        
        case "get_file_content":
            result = get_file_content(**keyword_arguments)
            return types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=function_name,
                        response={"result": result},
                    )
                ],
            )
        
        case "run_python_file":
            result = run_python_file(**keyword_arguments)
            return types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=function_name,
                        response={"result": result},
                    )
                ],
            )
        
        case "write_file":
            result = write_file(**keyword_arguments)
            return types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=function_name,
                        response={"result": result},
                    )
                ],
            )
        
        case _:
            return types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=function_name,
                        response={"error": f"Unknown function: {function_name}"},
                    )
                ],
            )


def main():
    if len(sys.argv) < 2:
        print("Invalid input!!!")
        sys.exit(1)
    messages = [
        types.Content(role="user", parts=[types.Part(text=sys.argv[1])]),
    ]
    available_functions = types.Tool(
        function_declarations=[
            schema_get_files_info,
            schema_get_file_content,
            schema_run_python_file,
            schema_write_file
        ]
    )
    system_prompt = """
    You are a helpful AI coding agent.

    When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

    - List files and directories
    - Read file contents
    - Execute Python files with optional arguments
    - Write or overwrite files

    All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.

    You will continue to iterate with added context until you get the answer.

    The result should be elaborated in points.
    """
    try:
        for i in range(20):
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=messages,
                config=types.GenerateContentConfig(
                    tools=[available_functions], 
                    system_instruction=system_prompt
                ),
            )
            
            if not response.function_calls and response.text:
                print(response.text)
                break
            
            # Add the assistant's response to messages
            for candidate in response.candidates:
                messages.append(candidate.content)

            for function_call_part in response.function_calls:
                function_result = call_function(function_call_part, verbose=(len(sys.argv) == 3 and sys.argv[2] == "--verbose"))
                messages.append(function_result)

                
    except Exception as e:
        print(f"Error: {e}")
        



if __name__ == "__main__":
    main()
