from fastapi import APIRouter
import google.generativeai as genai
import json

import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

prompt = """
    You are an AI that generates structured JSON output for a sequence of actions for an agent (id=0), based on the set goal. 
    You need to use the description of each object on the scene to get information about the scene and decide proper actions to reach the goal.

    Given the following goal: "{goal}"
    And the object descriptions (type, position and additional infos): {description}
    
    Generate a JSON response **strictly in this format**:
    {{
        "sequence": [
            {{
                "key": "move" or "speak" or "pickup" or "climb" or "shoot" or "consume"
                "action": {{
                    // Move: {{"targetId": <int>}}
                    // Speak: {{"dialog": "<str>"}}
                    // Pickup: {{"itemId": <int>}}
                    // Climb: {{"height": <float>}}
                    // Shoot: {{"targetId": <int>, "weapon": "<str>"}}
                    // Consume: {{"targetId": <int>}}
                }}
            }}
        ]
    }}

    - The "sequence" is a list containing multiple actions.
    - Each item in "sequence" has:
        - "key" as the action type ("move", "speak", "pickup", etc.)
        - "action" as an object with the required parameters.
    - Follows the schema, and does not include extra text.
    """

class LLM():
    def __init__(self, prompt: str):
        self.prompt = prompt
    
    def call(self, goal: str, desc: list) -> dict:
        try:
            response = model.generate_content(self.prompt.format(goal=goal, description=desc)).text
            # llm_data = json.loads(llm_response)

            return response
        except Exception as e:
            return {"error": f"LLM error: {str(e)}"}
        
llm_service = LLM()