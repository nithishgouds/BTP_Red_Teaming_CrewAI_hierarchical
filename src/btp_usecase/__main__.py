from .main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("btp_usecase.main:app", host="0.0.0.0", port=8000, reload=True)
