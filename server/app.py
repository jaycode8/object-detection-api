from fastapi import FastAPI, Request, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import io
from PIL import Image
from model.main import prediction

app = FastAPI()

templates = Jinja2Templates(directory="server/templates")
app.mount("/server/static", StaticFiles(directory="server/static"), name="static")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/detect")
async def test(image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    processed_image = prediction(image)
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type='image/png')

