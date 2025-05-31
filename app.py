from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the trained Keras model
model = load_model('model_checkpoint.h5')

app = FastAPI()


# pydantic model to validate incoming data
class UserInput(BaseModel):

    text: Annotated[str, Field(...,description='enter topic to joke about')]
    
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

@app.post('/predict')
def predict_joke(data: UserInput):
    text = data.text

    def sample_with_temperature(preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    for _ in range(10):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=38, padding='pre')
        pred = model.predict(padded_token_text, verbose=0)[0]
        pos = sample_with_temperature(pred, temperature=0.8)  

        predicted_word = "<unknown>"
        for word, index in tokenizer.word_index.items():
            if index == pos:
                predicted_word = word
                break

        text += " " + predicted_word

    return JSONResponse(status_code=200, content={'generated_joke': text})
