import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import action_gen_router

app = FastAPI(title="FastAPI Server")

origins = [
    "http://localhost:3000",
    "http://localhost:6789"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(action_gen_router, prefix="/rlhf")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=6789)