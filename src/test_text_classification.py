import text_classification as tc
import openai
import secrets_from_secretsmanager
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df_resume = pd.read_csv("resumes/resumes_test.csv")
text_embeddings = tc.generate_embeddings(
    df_resume["resume"], secrets_from_secretsmanager.get_secret()
)
text_embeddings_list = [
    text_embeddings[i].embedding for i in range(len(text_embeddings))
]

df_test = pd.DataFrame(text_embeddings_list, columns=tc.column_names)

# create target variable (sort of label for the data)
df_test["is_data_scientist"] = df_resume["role"] == "Data Scientist"

X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# model accuracy for test data
print(f"Model accuracy for test data: {tc.classifier.score(X_test, y_test)}")
# AUC value for test data
print(roc_auc_score(y_test, tc.classifier.predict_proba(X_test)[:, 1]))
