import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from app.main import app

    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")
