from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Store latest sensor data
latest_data = {"pir": None, "ir": None, "ultrasonic": None}

@app.get("/")
def home():
    return {"message": "ESP32 FastAPI Server is running successfully "}

# --- ESP32 posts data here ---
@app.post("/data")
async def receive_data(request: Request):
    global latest_data
    try:
        data = await request.json()
        latest_data.update(data)  # Save data
        print("Received from ESP32:", latest_data)
        return {"status": "success", "received": latest_data}
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=400)

# --- Streamlit (or user) can fetch it here ---
@app.get("/data")
def get_latest_data():
    if any(v is not None for v in latest_data.values()):
        return {"data": latest_data}
    else:
        return {"message": "No data received yet from ESP32."}
