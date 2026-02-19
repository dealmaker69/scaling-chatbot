# test_env.py
from dotenv import load_dotenv
import os

load_dotenv()           # This loads the .env file
key = os.getenv("OPENAI_API_KEY")

if key:
    print("SUCCESS! Your API key was found.")
    print("It starts with:", key[:10] + "...")   # just to confirm, not showing full key
else:
    print("FAIL — key not found. Check these:")
    print("1. Is the file really named .env (not .env.txt)?")
    print("2. Are you running this script from the same folder as .env?")
    print("3. Did you install python-dotenv? → pip install python-dotenv")