import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(".env")

API_KEY = os.getenv("OPENAGENDA_API_KEY")
AGENDA_UID = os.getenv("OPENAGENDA_AGENDA_UID", "56500817")

url = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"

response = requests.get(
    url,
    params={
        "key": API_KEY,
        "size": 1,
    },
    timeout=30
)

response.raise_for_status()

data = response.json()
event = data["events"][0]

print(event.keys())
print(json.dumps(event, indent=2, ensure_ascii=False))