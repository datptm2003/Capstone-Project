from pydantic import BaseModel
from typing import List, Dict

class ActionGenRequest(BaseModel):
    goal: str
    description: List[Dict]

class Action(BaseModel):
    key: str
    description: Dict