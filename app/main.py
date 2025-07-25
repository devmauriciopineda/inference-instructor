import instructor
import litellm
import boto3
import os
from pydantic import BaseModel
from litellm import completion
from config.config import settings

REGION = settings.aws_default_region
ACCESS_KEY = settings.aws_access_key_id
SECRET_KEY = settings.aws_secret_access_key

os.environ["LITELLM_LOG"] = "DEBUG"
litellm.api_base = "https://bedrock-runtime.us-east-1.amazonaws.com"

aws_client = boto3.client('bedrock-runtime', region_name=REGION)

AWS_MODEL = "bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0"


class User(BaseModel):
    name: str
    age: int


client = instructor.from_litellm(completion)


def query_model(messages):
    response = client.chat.completions.create(
        model=AWS_MODEL,
        api_base=litellm.api_base,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_region_name=REGION,
        max_tokens=1024,
        messages=messages,
        drop_params=True,
        response_model=User,
    )
    return response


text = "There was a boy in my class. People used to call him George and he had no more than 23 years old"
messages = [
    {
        "role": "user",
        "content": text
    }
]
user = query_model(messages)

# Print the user object
print(user)

# Print the attributes of the user object
print("Name: ", user.name)
print("Age: ", user.age)
