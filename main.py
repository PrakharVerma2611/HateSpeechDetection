import joblib

model = joblib.load('xgb_hate_speech_model.pkl')
vectorizer = joblib.load('xgb_tfidf_vectorizer.pkl')

def predict_hate(text):
    cleaned = text.lower()
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
    return label_map[label]

if __name__ == "__main__":
    while True:
        user_input = input("Enter a sentence (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        result = predict_hate(user_input)
        print(f"Prediction: {result}\n")
