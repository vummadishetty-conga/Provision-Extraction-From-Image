import os
import openai
import json
import re
import base64

# NOTE -  thsi is for the QA  enviroment
# os.environ['AZURE_OPENAI_API_KEY'] = 'a3ec8f5230094290ad74b27d0f8bb8a4'
# os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://genai-01.congacloud.io/qa'

os.environ['AZURE_OPENAI_API_KEY'] = 'da9e40054a6846e48acebc7887c29f5b'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://dev-aoai.azure-api.net'

#   AZURE_OPENAI_API_VERSION=2024-02-01 AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o AZURE_OPENAI_MINI_DEPLOYMENT_NAME=gpt-4o-mini-deployment-1100


# NOTE -  thsi is the old one for the dev enviroment
# os.environ['AZURE_OPENAI_API_KEY'] = 'b9f3441652304e89a0ead6c3092b2bdf'
# os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://dev-aoai.azure-api.net'
# Retrieve the environment variable

# Retrieve the API key and endpoint from environment variables
api_key = os.getenv('AZURE_OPENAI_API_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

openai.api_version = "2024-02-01"


def extract_provisions(image):
    prompt_text = """The image is a page from a contract document.
    Extract the following key terms 
    image and return it in json format
    Key Terms to be extracted:
        Agreement Title
        Company Signed Date
        Company Signed by Name
        Other Party (customer) Signed Date
        Other Party (customer) Signed  by Name
        Effective Date
    
    
    """

    # image_path = "TestImages/Test-avg.png"
    #
    # with open(image_path, "rb") as image_file:
    #     base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = image



    addtional_instrcution = """\n\n Special Instruction:  
    
                    The content where the value is extracted from should  belong to "payment term days" clause only.
                    calculate the confidence score of the content.
                    If the language is of a differnt clause then,  return extracted value as 'UNK'
    
                    """

    with_additional_instruction = prompt_text + addtional_instrcution

    response = openai.ChatCompletion.create(
        engine="gpt-4o",  # engine = "deployment_name".
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    )

    print(response)
    print(response['choices'][0]['message']['content'])