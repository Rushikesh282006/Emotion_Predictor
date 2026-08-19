from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from pathlib import Path
import pickle
import re
import numpy as np
import tensorflow as tf

# Get the absolute path to the project directory
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ARTIFACTS_DIR = BASE_DIR / "Artifacts"

def pad_sequences_manual(sequences, maxlen, padding='post', truncating='post', value=0):
    """
    A dependency-free implementation of Keras's pad_sequences.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'post':
                truncated_seq = seq[:maxlen]
            else: # pre
                truncated_seq = seq[len(seq) - maxlen:]
        else:
            truncated_seq = seq
        
        num_padding = maxlen - len(truncated_seq)
        if padding == 'post':
            padded_seq = truncated_seq + [value] * num_padding
        else: # pre
            padded_seq = [value] * num_padding + truncated_seq
            
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

# Loading the model and tokenizer once the server starts up
dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading TFLite Model and Tokenizer')
    model_file = ARTIFACTS_DIR / "Bidirec_gru_model.tflite"
    tokenizer_file = ARTIFACTS_DIR / "tokenizer.pkl"
    
    if not model_file.exists():
        raise FileNotFoundError(f"TFLite model file not found: {model_file}. Did you run convert_to_tflite.py?")
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
    
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()
    dl_model["interpreter"] = interpreter
    dl_model["input_details"] = interpreter.get_input_details()
    dl_model["output_details"] = interpreter.get_output_details()
    with open(str(tokenizer_file),'rb') as f:
        dl_model['Tokenizer'] = pickle.load(f)
    print("Model and server loaded successfully")
    yield
    dl_model.clear()

app = FastAPI(
    lifespan=lifespan,
)

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
    model_status = bool(dl_model.get("interpreter") and dl_model.get("Tokenizer"))
    return Health_Response(status="Server is Running", model_loaded=model_status)


@app.post('/predict',response_model = Prediction_Response)
def predict_response(Input_text : Input):
    interpreter = dl_model.get('interpreter')
    input_details = dl_model.get('input_details')
    output_details = dl_model.get('output_details')
    tokenizer = dl_model.get('Tokenizer')

    if not all([interpreter, input_details, output_details, tokenizer]):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later"
        )
    
    # Cleaning the text
    cleaned_text = preprocess(Input_text.text)

    # Convert the words into tokens
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])

    # Use the manual padding function
    padded_sequence = pad_sequences_manual(
        tokenized_text,
        maxlen=max_seq_len,
        padding="post",
        truncating="post"
    )

    # Prepare input tensor, ensuring correct dtype for the TFLite model
    input_data = np.array(padded_sequence, dtype=np.float32)

    # Set tensor, invoke, and get output
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_details[0]['index'])[0]

    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {label:float(prob) for label,prob in zip(emotion_labels,probabilities)}

    return Prediction_Response(
        text = Input_text.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_prob=all_probabilities
    )
