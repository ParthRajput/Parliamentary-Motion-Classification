import pandas as pd

df = pd.read_csv('/home/parth/Downloads/Coursework All/COMP-1804/comp1804_coursework_dataset_25-26_hansard_motions_full_revision.csv')

# FILTER FIRST (IMPORTANT)
selected_topics = [
    "Terrorism laws - For",
    "Reduce Spending on Welfare Benefits",
    "More powers for local councils",
    "Further devolution to Scotland",
    "Stop climate change",
    "Schools - Greater Autonomy"
]

df = df[df['motion_topic'].isin(selected_topics)]

print("Filtered distribution:\n", df['motion_topic'].value_counts())

# CLEANING
df = df.dropna(subset=['motion_text', 'motion_topic'])

# SPLIT
from sklearn.model_selection import train_test_split

x = df['motion_text']
y = df['motion_topic']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', ngram_range=(1,2), min_df=2)

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

#MODEL
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(x_train_tfidf, y_train)

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(x_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(x_train_tfidf, y_train)

import numpy as np

# Combine text + structured features
X_extra = df[['has_entity_date', 'has_entity_person', 'has_entity_event_or_org']].astype(int)

X_combined = np.hstack((x_train_tfidf.toarray(), X_extra.loc[x_train.index]))

df_sample = df.sample(100, random_state=42)
df_sample.to_excel("label_data.xlsx", index=False)