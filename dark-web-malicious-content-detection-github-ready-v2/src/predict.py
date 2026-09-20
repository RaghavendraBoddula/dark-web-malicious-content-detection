import re
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def predict_text_with_class(text, model, tokenizer, label_encoder, max_len=200):
    text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding="post")
    predicted_index = model.predict(padded_seq).argmax(axis=1)[0]
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_class

if __name__ == "__main__":
    model = load_model("textcnn_model.h5")
    tokenizer = joblib.load("textcnn_tokenizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    custom_text = input("Enter text to classify: ")
    print("Predicted Class:", predict_text_with_class(
        custom_text, model, tokenizer, label_encoder
    ))
