from sentence_transformers import SentenceTransformer
import sklearn
import numpy as np
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

app = FastAPI()
class DATA(BaseModel):
    text1: str
    text2: str

# using get request
@app.post("/enter_text")
async def enter_text(data: DATA):
    t1=data.text1
    t2=data.text2
    mt1=model.encode(t1)
    mt2=model.encode(t2)
    x=sklearn.metrics.pairwise.cosine_similarity(mt1.reshape(1,-1),mt2.reshape(1,-1))
    x= "{:.2f}".format(float(x))
    return {
    "similarity score": x
    }
