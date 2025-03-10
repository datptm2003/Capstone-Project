from fastapi import APIRouter, HTTPException
import json
from fastapi.responses import JSONResponse

from app.model import ActionGenRequest, Action
from app.services import llm_service

action_gen_router = APIRouter()

@action_gen_router.post('/gen-actions')
def gen_actions(req: ActionGenRequest):
    llm_response = llm_service.call(req.goal, req.description)

    # TODO: Add inference logic

