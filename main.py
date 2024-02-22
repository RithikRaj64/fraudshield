from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import text, img, vid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_origins=["*"],
    allow_headers=["*"],
)

@app.get("/api/text")
def chk_text(query : str):
    res = text(query)
    return {"Response" : res}

@app.get("/api/vid")
def chk_vid(link : str):
    res = vid(link);
    return {"Response" : res}

@app.get("/api/img")
def chk_img(link : str):
    res = img(link)
    return {"Response" : res}

