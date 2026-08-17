from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
import pickle
import re
from pathlib import Path
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Get the absolute path to the project directory
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ARTIFACTS_DIR = BASE_DIR / "Artifacts"

# Loading the model and tokenizer once the server starts up
dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading Model and Tokenizer')
    model_file = ARTIFACTS_DIR / "Bidirec_gru_model.keras"
    tokenizer_file = ARTIFACTS_DIR / "tokenizer.pkl"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
    
    dl_model["BIGRU Model"] = load_model(str(model_file))
    with open(str(tokenizer_file),'rb') as f:
        dl_model['Tokenizer'] = pickle.load(f)
    print("Model and server loaded successfully")
    yield #Pause : Model is loaded and server is running 
    dl_model.clear()

app = FastAPI(
    lifespan=lifespan,
)

model_path = str(ARTIFACTS_DIR / "Bidirec_gru_model.keras")
tokenizer_path = str(ARTIFACTS_DIR / "tokenizer.pkl")

max_seq_len = 50

emotion_labels = ['sadness','joy','love','anger','fear','surprise']

emotion_emojis = {
    'sadness': '😞',
    'joy': '😆',
    'love': '😍',
    'anger': '😡',
    'fear': '😱',
    'surprise': '😮'
}

"""
Converting string to lowercase
removing extra whitespaces
removing punctuations and special characters

"""
def preprocess(text: str) -> str:
    text = text.lower()
    # Handle negations
    
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Input(BaseModel):
    text : str = Field(
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example" : "I am feeling good"}
    )

class Prediction_Response(BaseModel):
    text : str
    predicted_emotion : str
    confidence : float
    all_prob : dict[str,float]


class Health_Response(BaseModel):
    status: str
    model_loaded  : bool

"""
 Mount the static files to the fastapi
 Enable CORS to allow requests from diff server
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Mount static files with absolute path
if not STATIC_DIR.exists():
    raise FileNotFoundError(f"Static directory not found: {STATIC_DIR}")
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)))


"""
API Endpoints
1.Server UI at homepage ('/')
2.Health Check Endpoints ('/health')
3.Predict Emotion Endpoint ('/predict')
"""

@app.get('/', include_in_schema = False)
def server_ui():
    index_file = STATIC_DIR / 'index.html'
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    return FileResponse(str(index_file))



@app.get('/health', response_model=Health_Response)
def health_check():
    model_status = bool(dl_model.get("BIGRU Model") and dl_model.get("Tokenizer"))
    return Health_Response(status="Server is Running", model_loaded=model_status)


@app.post('/predict',response_model = Prediction_Response)
def predict_response(Input_text : Input):
    BiGRU_Model = dl_model.get('BIGRU Model')
    tokenizer = dl_model.get('Tokenizer')

    if BiGRU_Model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later"
        )
    
    # Cleaning the text
    cleaned_text = preprocess(Input_text.text)

    # Convert the words into tokens
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
    tokenized_text,
    maxlen = max_seq_len,
    padding ="post",
    truncating="post"
    )


    probabilities = BiGRU_Model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {label:float(prob) for label,prob in zip(emotion_labels,probabilities)}

    return Prediction_Response(
        text = Input_text.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_prob=all_probabilities
    )

