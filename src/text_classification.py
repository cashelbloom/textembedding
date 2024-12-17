import openai
import secrets_from_secretsmanager
import xlsxwriter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df_resume = pd.read_csv("resumes/resumes_train.csv")


def generate_embeddings(text, openai_api_key):
    openai_client = openai.OpenAI(api_key=openai_api_key)

    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data


openai_api_key = secrets_from_secretsmanager.get_secret()
text_embeddings = generate_embeddings(df_resume["resume"], openai_api_key)
text_embeddings_list = [
    text_embeddings[i].embedding for i in range(len(text_embeddings))
]
# print(f"contents of text_embeddings_list: {text_embeddings_list}")
# text_embeddings_df = pd.DataFrame(text_embeddings_list)
# pd.ExcelWriter("resumes/resumes_train_embeddings.xlsx", engine="xlsxwriter")
# text_embeddings_df.to_excel("resumes/resumes_train_embeddings.xlsx")
#
column_names = ["embedding_" + str(i) for i in range(len(text_embeddings_list[0]))]

df_train = pd.DataFrame(text_embeddings_list, columns=column_names)
df_train["is_data_scientist"] = df_resume["role"] == "Data Scientist"

X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

classifier = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)

# # print the model accuracy
print(f"Model accuracy: {classifier.score(X, y)}")

# # AUC value for training data
print(roc_auc_score(y, classifier.predict_proba(X)[:, 1]))
