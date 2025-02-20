# import requests
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()

def main():
    configure()
    elsevier_api_key = os.getenv('elsevier_api_key')
    print(elsevier_api_key)

main()
