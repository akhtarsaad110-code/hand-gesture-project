import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# Model load karna (Ensure karein ke gesture_model.h5 'model' folder ke andar ho)
MODEL_PATH = "model/gesture_model.h5"
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

# Sign Language MNIST Labels (A to Y)
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>Sign Language DL Project</title></head>
        <body style="font-family: Arial; text-align: center; margin-top: 50px; background-color: #f4f4f9;">
            <div style="background: white; max-width: 600px; margin: 0 auto; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
                <h1>Sign Language Classifier 🤟</h1>
                <p>Upload a hand gesture image (28x28 grayscale) to predict the letter.</p>
                <form action="/predict" enctype="multipart/form-data" method="post" style="margin-top: 20px;">
                    <input name="file" type="file" required style="margin-bottom: 10px;"><br>
                    <input type="submit" value="Predict Gesture" style="padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;">
                </form>
                
                <hr style="margin-top: 30px; border: 0; border-top: 1px solid #ddd;">
                <div style="text-align: left; background: #eee; padding: 15px; border-radius: 5px; margin-top: 20px;">
                    <h3>📌 Project Links (For Automated Evaluation)</h3>
                    <ul>
                        <li>GitHub Repo: <a href="YOUR_GITHUB_URL" target="_blank">Repository Link</a></li>
                        <li>Project Report: <a href="YOUR_REPORT_URL" target="_blank">View Report</a></li>
                        <li>YouTube Demo: <a href="YOUR_YOUTUBE_URL" target="_blank">Watch Video</a></li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model file not found on server."}
    
    contents = await file.read()
    # Sign Language MNIST ke mutabiq image ko Grayscale aur 28x28 size me convert karna
    img = Image.open(io.BytesIO(contents)).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0])) * 100
    
    letter = CLASSES[pred_idx] if pred_idx < len(CLASSES) else str(pred_idx)
    return {
        "Prediction": letter, 
        "Confidence": f"{confidence:.2f}%"
    }