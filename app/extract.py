import instructor
import litellm
from pydantic import BaseModel
from litellm import completion
from config.config import settings
from pypdf import PdfReader
import requests

REGION = settings.aws_default_region
ACCESS_KEY = settings.aws_access_key_id
SECRET_KEY = settings.aws_secret_access_key
EXCHANGE_RATES_API_KEY = settings.exchange_rates_api_key
AWS_MODEL = "bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0"
URL_EXCHANGE = "https://v6.exchangerate-api.com/v6"
litellm.api_base = "https://bedrock-runtime.us-east-1.amazonaws.com"
client = instructor.from_litellm(completion)


class Budget(BaseModel):
    """Budget model."""
    value: float
    currency: str


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def query_model(messages):
    """Query the LLM model with the provided messages."""
    response = client.chat.completions.create(
        model=AWS_MODEL,
        api_base=litellm.api_base,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_region_name=REGION,
        max_tokens=100,
        messages=messages,
        drop_params=True,
        response_model=Budget,
    )
    return response


def convert_currency(value, from_currency, to_currency):
    """Convert currency from one type to another using the exchange rates API."""
    if from_currency == to_currency:
        return Budget(value=value, currency=to_currency)

    # Make a request to the exchange rates API
    result = requests.get(f"{URL_EXCHANGE}/{EXCHANGE_RATES_API_KEY}/latest/{from_currency}").json()
    rate = result.get("conversion_rates").get(to_currency)
    print(f"Rate: {rate}")
    converted_budget = Budget(value=rate * value, currency=to_currency)
    return converted_budget


# Example usage
file_path = "data/notification.pdf"
text = extract_text_from_pdf(file_path)

# Build the prompt for the LLM
prompt = (
    "Extract the budget from the following text. ",
    "Give the amount and the currency in a json format:\n\n",
    f"\n\nText: {text}\n\n"
)
messages = [
    {
        "role": "user",
        "content": prompt
    }
]

# Query the model with the extracted text
print("Querying the model...")
budget = query_model(messages)

# Print the budget object
print(budget)

# Convert the budget to different currencies
target_currencies = ["USD", "COP"]

for c in target_currencies:
    print(f"Converting to {c}")
    converted_budget = convert_currency(budget.value, budget.currency, c)
    print(converted_budget)
