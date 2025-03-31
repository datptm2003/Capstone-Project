from fastapi import APIRouter
import google.generativeai as genai
import json

from prompts import llm_prompt

import re
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


class LLM():
    def __init__(self, prompt: str):
        self.prompt = prompt
    
    def call(self, scene: list, goal: str, actions: list, feedback: str) -> dict:
        try:
            response = model.generate_content(self.prompt.format(goal=goal, scene=scene, actions=actions, feedback=feedback)).text
            # llm_data = json.loads(llm_response)
            match = re.search(r'\{.*\}', response, re.DOTALL)

            if match:
                response = match.group(0)
            # print("---------------------------------------------------------------------")
            # print(response)
            # print("88888888888888888888888888888888888888")
            print((json.loads(response))['sequence'])

            return (json.loads(response))['sequence']
        except Exception as e:
            return {"error": f"LLM error: {str(e)}"}
        
llm_service = LLM(llm_prompt)