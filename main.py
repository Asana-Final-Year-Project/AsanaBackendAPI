import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile, File
from tempfile import NamedTemporaryFile
import aiofiles
import os

from fastapi.middleware.cors import CORSMiddleware

import tasks

app = FastAPI()
    
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    

@app.get("/hello")
def return_greetings():
    return {"msg" : "Hello!"}

@app.post("/video")
async def classify_video(file: UploadFile = File(...)):
    if not file.filename.endswith(".mp4"):
        return {"message": "Only MP4 videos are supported"}
    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            try:
                contents = await file.read()
                await temp.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                await file.close()
        prediction = tasks.prediction_main(temp.name)
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)
    return {"Pose" : prediction}
