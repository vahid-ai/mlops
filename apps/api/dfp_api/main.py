from fastapi import FastAPI

app = FastAPI(title="DFP API")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
