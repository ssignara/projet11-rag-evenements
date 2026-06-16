import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAGENDA_API_KEY")

response = requests.get(
    "https://api.openagenda.com/v2/agendas",
    params={
        "key": API_KEY,
        "search": "ile de france"
    }
)

print(response.status_code)
print(response.json())