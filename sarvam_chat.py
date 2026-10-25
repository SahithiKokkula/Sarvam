import os

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise SystemExit("Add your Sarvam API key to .env before running this script.")

client = SarvamAI(api_subscription_key=api_key)
response = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[{"role": "user", "content": "Hello! Reply with one short greeting."}],
)
print(response.choices[0].message.content)
