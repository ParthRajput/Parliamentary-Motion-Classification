import pandas as pd

df = pd.read_csv('/home/parth/Downloads/Coursework All/COMP-1804/comp1804_coursework_dataset_25-26_hansard_motions_full_revision.csv')

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst rows:\n", df.head())
print("\nInfo:\n")
print(df.info())

#Missing values
print("\nMissing values:\n", df.isnull().sum())

#Target Distribution
print("\nMotion topic distribution:\n", df['motion_topic'].value_counts())

#Check text length
df['text_length'] = df['motion_text'].apply(lambda x : len(str(x)))
print("\nText length stats:\n", df['text_length'].describe())

df = df.dropna(subset=['motion_text', 'motion_topic'])

from sklearn.model_selection import train_test_split

x = df['motion_text']
y = df['motion_topic']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(x_train_tfidf, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(x_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

selected_topics = [
    "Terrorism laws - For",
    "Reduce Spending on Welfare Benefits",
    "More powers for local councils",
    "Further devolution to Scotland",
    "Stop climate change",
    "Schools - Greater Autonomy"
]

df_filtered = df[df['motion_topic'].isin(selected_topics)]
df_filtered['motion_topic'].value_counts()

