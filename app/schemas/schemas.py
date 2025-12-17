from pydantic import BaseModel



class PromptRequest(BaseModel):
    description: str

class ImageRequest(BaseModel):
    prompt: str
