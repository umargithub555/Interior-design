from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import ZImagePipeline
import torch
import google.generativeai as genai
import os
from dotenv import load_dotenv
import uuid
from app.routers import image_creation_router


app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.get("/")
# def read_root():
#     return {"status": "Interior Design AI API Running"}


app.include_router(image_creation_router.router)




@app.get("/health")
def health_check():
    return {"status":"running"}





@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("app/static/frontend.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found in static folder</h1>"
