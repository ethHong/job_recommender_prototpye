from flask import render_template, request, jsonify, session, Flask
from flask_session import Session
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

import pickle
import os
from ast import literal_eval
import numpy as np
import pandas as pd
from stopword_filter import (
    filter_stopwords,
    cleanse,
    sw,
    divterms,
    divcats,
    tfidf_matrix,
    tfidf,
)
from sklearn.preprocessing import MinMaxScaler
from LDAModel import train_lda, jsd, KL
import numpy as np

from commands import create_tables
from extensions import db
from models import user_rating

from tika import parser

# Test Alpha Value, alpha=1 on TFIDF
alpha = 0.5

app = Flask(__name__)
app.debug = True
app.config.from_pyfile("settings.py")
db.init_app(app)
Session(app)
CORS(app)

UPLOAD_FOLDER = os.getcwd() + "/app/USERCVs"
ALLOWED_EXTENSIONS = set(["pdf"])
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def predict():

    input_val = [str(x) for x in request.form.values()]
    # Procss PDF
    print("Checkup:", input_val)
    username = input_val[0]
    email = input_val[1]

    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS

    print("saving file...")
    file = request.files["file"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    pdf_file = os.getcwd() + "/app/USERCVs/" + filename

    raw = parser.from_file(pdf_file)
    text_raw = raw["content"]

    session["user"] = username
    session["email"] = email
    session["CV"] = text_raw

    target = text_raw

    def process_filter(target):

        target = filter_stopwords(target, sw)
        target = cleanse(target)
        target = target.split()

        return target

    target_doc = process_filter(target)

    # Check if there's no result
    if len(target) == 0:
        return render_template("error.html")

    else:

        data = pd.read_csv(os.getcwd() + "/app/train_dataset/Database.csv")
        data["tokenized"] = data["tokenized"].apply(literal_eval)

        # 미리 로드한 모델 - 모든 수업으로 LDA모델링
        dictionary = pickle.load(
            open(os.getcwd() + "/app/model/FIND_dictionary.pkl", "rb")
        )
        corpus = pickle.load(open(os.getcwd() + "/app/model/FIND_corpus.pkl", "rb"))
        lda = pickle.load(open(os.getcwd() + "/app/model/FIND_model.pkl", "rb"))

        doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
        query_bow = dictionary.doc2bow(target_doc)
        new_doc_distribution = np.array(
            [tup[1] for tup in lda.get_document_topics(bow=query_bow)]
        )

        index_to_score_JSD = jsd(new_doc_distribution, doc_topic_dist)
        index_to_score_KL = KL(new_doc_distribution, doc_topic_dist)

        data["distance_with_JD_JSD"] = index_to_score_JSD
        data["distance_with_KL"] = index_to_score_KL

        # min_max_scaler = MinMaxScaler()
        # min_max_scaler.fit(data["distance_with_JD_JSD"].values.reshape(-1, 1))
        # scaled_JSD = min_max_scaler.transform(data["distance_with_JD_JSD"].values.reshape(-1, 1))

        # min_max_scaler.fit(data["distance_with_KL"].values.reshape(-1, 1))
        # scaled_KL = min_max_scaler.transform(data["distance_with_KL"].values.reshape(-1, 1))

        weight = alpha
        k = 5
        # TF-IDF 스코어
        jd_tfidf = tfidf(target)
        jd_keywords = jd_tfidf["words"].values[:k]  # Expanded Keyword
        tf_matrix = tfidf_matrix(data)
        vector = np.array([0] * tf_matrix.shape[1])

        for i in jd_keywords:
            if i in tf_matrix.index:
                vector = vector + tfidf_matrix(data).loc[i].values

        data["TF-IDF_keyscore"] = vector
        data["distance_with_JD_JSD"] = data["distance_with_JD_JSD"].apply(
            lambda x: 1 / (x + 0.001)
        )

        # minmax
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(data["TF-IDF_keyscore"].values.reshape(-1, 1))
        data["TF-IDF_keyscore"] = min_max_scaler.transform(
            data["TF-IDF_keyscore"].values.reshape(-1, 1)
        )

        min_max_scaler.fit(data["distance_with_JD_JSD"].values.reshape(-1, 1))
        data["distance_with_JD_JSD"] = min_max_scaler.transform(
            data["distance_with_JD_JSD"].values.reshape(-1, 1)
        )

        # data["distance_with_KL"] = data["distance_with_KL"].apply(lambda x: 1 / (x + 0.001))
        data["Aggregate_score"] = (
            data["distance_with_JD_JSD"] * (1 - weight)
            + data["TF-IDF_keyscore"] * weight
        )
        result = data[
            ["Position", "Job_Details", "tokenized", "Aggregate_score"]
        ].sort_values(by="Aggregate_score", ascending=False)
        result = result[:10]
        result = result.reset_index(drop=True)

        output = result[["Position", "Job_Details", "Aggregate_score"]]
        output.columns = ["Position", "Job_Details", "Aggregate_score"]

        print("Checkup1: ", output)

        min_max_scaler = MinMaxScaler((0, 10))
        min_max_scaler.fit(output["Aggregate_score"].values.reshape(-1, 1))
        scaled_score1 = min_max_scaler.transform(
            output["Aggregate_score"].values.reshape(-1, 1)
        )
        output["Aggregate_score"] = scaled_score1.round(2)

        output = output.drop(["Aggregate_score"], axis=1)

        results = []
        for i in range(0, output.shape[0]):
            results.append(dict(output.iloc[i]))

        session["rating"] = results

    return render_template("result.html", result_1=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
