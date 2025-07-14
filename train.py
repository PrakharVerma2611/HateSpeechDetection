import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()
    return text


df = pd.read_csv('archive/data/labeled_data.csv')[['class', 'tweet']]
df.columns = ['label', 'text']
df['text'] = df['text'].apply(clean_text)

print("Label distribution:\n", df['label'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=3,
    max_depth=6,
    learning_rate=0.2,
    n_estimators=200
)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Hate Speech", "Offensive", "Neutral"]))


joblib.dump(model, 'xgb_hate_speech_model.pkl')
joblib.dump(vectorizer, 'xgb_tfidf_vectorizer.pkl')

