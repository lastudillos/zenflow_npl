from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model import LicitacionesModel

app = FastAPI()
model = LicitacionesModel()

class Preferences(BaseModel):
    preferences: List[str]

@app.post("/licitations/")
async def get_licitations(preferences: Preferences):
    try:
        results = model.find_compatible_licitations(preferences.preferences)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
