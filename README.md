# Emotion Scanner

A sophisticated web application that analyzes text to detect and classify emotions using a bidirectional GRU (Bi-GRU) neural network model. The application provides real-time emotion detection with confidence scores and probability distributions across six emotion categories.

## 🎯 Features

- **Real-time Emotion Detection**: Analyze text and instantly identify the underlying emotion
- **Six Emotion Classes**: Detects sadness, joy, love, anger, fear, and surprise
- **Confidence Scoring**: Provides confidence percentage for the predicted emotion
- **Probability Distribution**: Displays probability breakdown across all emotion categories
- **Interactive UI**: Modern instrument-panel inspired design with smooth animations
- **Model Status Monitoring**: Real-time health checks for model availability
- **CORS Enabled**: Secure cross-origin requests support
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🏗️ Technical Stack

- **Backend**: FastAPI (Python)
- **Deep Learning**: TensorFlow/Keras
- **Model Architecture**: Bidirectional GRU (Bi-GRU) Neural Network
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Web Server**: Uvicorn
- **Python Version**: 3.11.9

## 📁 Project Structure

```
Semantic_Analyzer_project/
├── main.py                          # FastAPI application and API endpoints
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python runtime specification
├── README.md                        # This file
├── Artifacts/
│   ├── Bidirec_gru_model.keras     # Pre-trained Bi-GRU model
│   └── tokenizer.pkl                # Trained tokenizer for text preprocessing
└── static/
    ├── index.html                   # Frontend UI
    ├── script.js                    # Client-side logic and API integration
    └── style.css                    # Styling and animations
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11.9 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
cd Semantic_Analyzer_project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI web server
- **tensorflow-cpu**: Machine learning framework (CPU version)
- **numpy**: Numerical computing library
- **pydantic**: Data validation
- **python-multipart**: Form data handling

### Step 3: Verify Model Files

Ensure the following files exist in the `Artifacts/` directory:
- `Bidirec_gru_model.keras` - The trained emotion detection model
- `tokenizer.pkl` - The text tokenizer

### Step 4: Run the Application

```bash
uvicorn main:app --reload --host 0.0.0.0
```

The application will start and display:
```
Uvicorn running on http://127.0.0.1:8000
```

Open your browser and navigate to `http://localhost:8000` to access the application.

## 📖 Usage

### Web Interface

1. **Enter Text**: Type or paste a sentence in the input area (max 2000 characters)
2. **Scan**: Click the "Scan text" button or press `Ctrl/Cmd + Enter`
3. **View Results**:
   - **Top Emotion**: Displayed with corresponding emoji
   - **Confidence**: Percentage confidence in the prediction
   - **Probability Spectrum**: Bar chart showing probabilities for all emotions

### API Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "Server is Running",
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK`: Server is running and model is loaded
- `503 Service Unavailable`: Model is still loading

---

#### 2. Emotion Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "I am feeling good today"
}
```

**Request Parameters:**
- `text` (string, required): The text to analyze
  - Minimum length: 1 character
  - Maximum length: 2000 characters

**Response:**
```json
{
  "text": "I am feeling good today",
  "predicted_emotion": "joy",
  "confidence": 0.9847,
  "all_prob": {
    "sadness": 0.0023,
    "joy": 0.9847,
    "love": 0.0089,
    "anger": 0.0012,
    "fear": 0.0015,
    "surprise": 0.0014
  }
}
```

**Response Fields:**
- `text`: Original input text
- `predicted_emotion`: Detected emotion (one of: sadness, joy, love, anger, fear, surprise)
- `confidence`: Confidence score as a float (0.0 - 1.0)
- `all_prob`: Probability distribution across all emotion categories

**Error Response (Model Not Loaded):**
```
Status: 503 Service Unavailable

{
  "detail": "Model not loaded yet. Please try again later"
}
```

---

#### 3. Serve Frontend UI
```http
GET /
```

Returns the interactive web interface (index.html)

---

#### 4. Serve Static Files
```
GET /static/style.css
GET /static/script.js
```

Serves CSS and JavaScript resources

## 🧠 Model Details

### Architecture
- **Type**: Bidirectional Gated Recurrent Unit (Bi-GRU)
- **Input**: Tokenized and padded text sequences
- **Sequence Length**: 50 tokens (maximum)
- **Output**: 6 emotion classes with probability distribution

### Emotion Classes
| Emotion | Emoji | Color |
|---------|-------|-------|
| Sadness | 😞 | Blue (#5C86C9) |
| Joy | 😆 | Yellow (#F2B705) |
| Love | 😍 | Pink (#EF6F92) |
| Anger | 😡 | Red (#E1483F) |
| Fear | 😱 | Purple (#9569D6) |
| Surprise | 😮 | Teal (#2FBF9F) |

### Text Preprocessing Pipeline
1. **Lowercase Conversion**: Convert all text to lowercase
2. **Remove Apostrophes**: Strip single quotes
3. **Remove Special Characters**: Keep only alphanumeric characters and spaces
4. **Normalize Whitespace**: Remove extra spaces and trim

### Model Files
- **Model File**: `Bidirec_gru_model.keras` (~12 MB)
  - Format: Keras H5 format
  - Pre-trained on emotion dataset
  
- **Tokenizer File**: `tokenizer.pkl` 
  - Format: Pickle serialization
  - Contains vocabulary and word-to-index mappings

## 🔧 Configuration

Edit these values in `main.py` to customize behavior:

```python
max_seq_len = 50              # Maximum sequence length for padding
emotion_labels = [...]        # List of emotion categories
emotion_emojis = {...}        # Emoji mappings
```

### CORS Settings
```python
allow_origins = ["*"]         # Allow requests from any origin
allow_credentials = True      # Allow credentials
allow_methods = ["*"]         # Allow all HTTP methods
allow_headers = ["*"]         # Allow all headers
```

## 📊 Example Requests

### Using cURL
```bash
# Health check
curl http://localhost:8000/health

# Emotion prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this project!"}'
```

### Using Python Requests
```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Emotion prediction
payload = {"text": "I love this project!"}
response = requests.post('http://localhost:8000/predict', json=payload)
print(response.json())
```

### Using JavaScript Fetch
```javascript
// Health check
fetch('/health')
  .then(res => res.json())
  .then(data => console.log(data))

// Emotion prediction
fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'I love this project!' })
})
  .then(res => res.json())
  .then(data => console.log(data))
```

## 🎨 UI Features

### Design System
- **Color Scheme**: Dark mode with emotion-based color accents
- **Typography**: Space Grotesk (display), Inter (body), IBM Plex Mono (code)
- **Animations**: Smooth transitions and scanning effects
- **Responsive**: Mobile-first design approach

### Key Components
1. **Status Indicator**: Shows model loading status
2. **Input Area**: Text textarea with character counter
3. **Scan Button**: Triggers emotion detection
4. **Result Display**: Shows emoji, emotion label, and confidence
5. **Probability Spectrum**: Visual representation of all emotion probabilities

## ⚠️ Troubleshooting

### Model Not Loading
```
Error: "Model not loaded yet. Please try again later"
```
**Solution**: Wait a few seconds after starting the server. The model loads during startup.

### Import Errors
```
Error: ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

### Model File Not Found
```
Error: FileNotFoundError: Artifacts/Bidirec_gru_model.keras
```
**Solution**: Ensure model files exist in the `Artifacts/` directory.

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Use a different port:
```bash
uvicorn main:app --port 8001
```

## 📝 License

This project is provided as-is for educational and research purposes.

## 👨‍💻 Author

Created as a demonstration of emotion detection using deep learning and modern web technologies.

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

## 📞 Support

For issues or questions, please refer to the project documentation or examine the source code comments.

---

**Last Updated**: August 2026  
**Python Version**: 3.11.9  
**Framework**: FastAPI 0.115.6
