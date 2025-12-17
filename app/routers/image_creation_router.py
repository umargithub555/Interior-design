import torch
from app.schemas.schemas import PromptRequest, ImageRequest
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uuid
import os
from dotenv import load_dotenv
import google.generativeai as genai
from diffusers import ZImagePipeline



load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



router = APIRouter()





# Load model ONCE at startup
MODEL_PATH = "ZImageModel"
output_dir = "app/images"

os.makedirs(output_dir, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
pipe = ZImagePipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False
)
pipe.to(device)
print("Model loaded successfully!")

gemini_model = genai.GenerativeModel("gemini-2.0-flash")

SYSTEM_PROMPT = """
You are a 'Prompt Architect' for an interior design generative AI system.
Your job is to take the user's messy natural language interior description
and convert it into a clean, cinematic, professional, highly visual prompt
for an image generation model.

If the prompt is about something else apart from interior design do make it better for an image model as well.

Rules:
- Just give the plain prompt and not any additional explanation.
- Add missing details that help AI imagine the scene.
- Add scene descriptors: photorealistic, wide-angle, 4K, detailed textures.
- Include placement logic (e.g., sofa against long wall).
- Add matching lighting, mood, atmosphere.
- Keep it realistic and grounded.
"""










@router.post("/generate-prompt")
def generate_prompt(request: PromptRequest):
    try:
        response = gemini_model.generate_content(
            SYSTEM_PROMPT + "\n\nUser description:\n" + request.description
        )
        return {"prompt": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/generate-image")
def generate_image(request: ImageRequest):
    try:
        # Generate unique filename
        filename = f"generated_{uuid.uuid4()}.png"

        filepath = os.path.join(output_dir, filename)
        
        # Generate image
        image = pipe(
            prompt=request.prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]
        
        image.save(filepath)
        
        return {"image_path": filepath, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/image/{file_path:path}")
def get_image(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")
