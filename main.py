from utils import load_model, get_class_names, get_transform, get_video_as_inputs, get_preds, get_counts
from fastapi import FastAPI, File, UploadFile
from typing import Optional
from pydantic import BaseModel


model, device = load_model()


app = FastAPI()


# @app.post("/files/")
# async def create_file(file: bytes = File(...)):
#     return {"file_size": len(file)}

def get_prediction(path):
    class_names = get_class_names()
    transform = get_transform()
    inputs = get_video_as_inputs(path, transform, device)
    preds = get_preds(inputs, model, class_names)

    return preds


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    with open("current.mp4", "wb") as f:
        f.write(contents)

    preds = get_prediction("current.mp4")

    counts, calories = get_counts(preds[0], "current.mp4")
    return {"predictions": preds, "calories_burnt": calories, "reps": counts}
