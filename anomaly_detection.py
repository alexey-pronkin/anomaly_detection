import time

import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from nltk.stem.snowball import RussianStemmer

# from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer,
)
from sklearn.preprocessing import (
    StandardScaler,
    # LabelEncoder,
    OneHotEncoder,
    PolynomialFeatures,
)
from sklearn.compose import ColumnTransformer

# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import FeatureUnion
# from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold

import logging

# logger = logging.getLogger(__name__)
# change for prints in case of docker

print("Starting Exploratory Data Analysis (EDA)")
ueba = pd.read_csv("ueba.csv", index_col=0)
ueba.describe()
print("Cheking unique domains, ids, uid")
print(
    f"unique domains = {set(ueba.domain)}, number of unique ids = {len(set(ueba.id))}, number of unique uids {len(set(ueba.uid))}"
)
print("NaN in uid gives us first three anomaly persons:")
anomalies_1 = ueba[ueba.uid.isna()]
print(anomalies_1)
print("Let's count by aggregating lenght of groups ids:")
ueba["len_member_of_groups_ids"] = ueba["member_of_groups_ids"].apply(
    lambda x: len(str(x))
)
ueba.groupby(by="len_member_of_groups_ids").count()
print(
    "Future investigation in NANs and grops ids gives us some suspicious people with small logins and huge rights:"
)
anomalies_2 = ueba[
    np.any(
        np.c_[
            (ueba.len_member_of_groups_ids == 267).values,
            (ueba.len_member_of_groups_ids == 119).values,
            # (ueba.len_member_of_groups_ids==115).values,
        ],
        axis=1,
    )
]
print(anomalies_2)
print(
    "Future investigation in NANs and grops ids gives us some suspicious people with huge rights, we need to check them:"
)
print("Let's prepare data and try some ML algorithms:")
data = ueba.drop(["domain"], axis=1)
data["text"] = data["cn"] + " " + data["title"] + " " + data["who"]
categorical_cols = ["department"]
data.drop(["cn", "title", "who"], axis=1)
data.drop(["id", "uid"], axis=1, inplace=True)
print("Filling NaNs with simple method:")
data[
    [
        "cn",
        "department",
        "title",
        "who",
        "text",
        "member_of_indirect_groups_ids",
        "member_of_groups_ids",
    ]
] = data[
    [
        "cn",
        "department",
        "title",
        "who",
        "text",
        "member_of_indirect_groups_ids",
        "member_of_groups_ids",
    ]
].fillna(
    value="NAN"
)
data.fillna(value="-1", inplace=True)
print("Defining data preprocessing:")
# Some code taken from:
# https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py


stemmer = RussianStemmer()
analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


# Settings
n_samples = 2633
n_outliers = 5
outliers_fraction = n_outliers / n_samples
n_inliers = n_samples - n_outliers
categorical_cols = ["cn", "department", "title", "who"]
transform_algorithms = [
    (
        "Features baseline",
        make_pipeline(
            ColumnTransformer(
                [
                    (
                        "OHE",
                        OneHotEncoder(
                            sparse_output=True, handle_unknown="infrequent_if_exist"
                        ),
                        categorical_cols,
                    ),
                    # handle categorical columns as one hot encoding
                    (
                        "Text",
                        HashingVectorizer(
                            ngram_range=(3, 6), analyzer="char_wb", n_features=20000
                        ),
                        "text",
                    ),
                    *[
                        (
                            col,
                            TfidfVectorizer(
                                ngram_range=(1, 4),
                                preprocessor=lambda x: " ".join(x.split(";")),
                                analyzer="word",
                            ),
                            col,
                        )
                        for col in [
                            "member_of_indirect_groups_ids",
                            "member_of_groups_ids",
                        ]
                    ],  # handle categories as text, interaction up to 4 included
                ]
            ),
            VarianceThreshold(threshold=0.0001),
            PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            ),  # intersting interaction features like Is_admin * title
            VarianceThreshold(),  # remove zero varience features
            StandardScaler(
                with_mean=False
            ),  # Scaling is important for anomaly detection
        ),
    ),
    (
        "Features without scaling",
        make_pipeline(
            ColumnTransformer(
                [
                    (
                        "OHE",
                        OneHotEncoder(
                            sparse_output=True, handle_unknown="infrequent_if_exist"
                        ),
                        categorical_cols,
                    ),
                    # handle categorical columns as one hot encoding
                    (
                        "Text",
                        HashingVectorizer(
                            ngram_range=(3, 6), analyzer="char_wb", n_features=20000
                        ),
                        "text",
                    ),
                    *[
                        (
                            col,
                            TfidfVectorizer(
                                ngram_range=(1, 4),
                                preprocessor=lambda x: " ".join(x.split(";")),
                                analyzer="word",
                            ),
                            col,
                        )
                        for col in [
                            "member_of_indirect_groups_ids",
                            "member_of_groups_ids",
                        ]
                    ],  # handle categories as text, interaction up to 4 included
                ]
            ),
            VarianceThreshold(threshold=0.0001),
            PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            ),  # intersting interaction features like Is_admin * title
            VarianceThreshold(),  # remove zero varience features
        ),
    ),
    (
        "Features CountVectorizer with Stemmer",
        make_pipeline(
            ColumnTransformer(
                [
                    (
                        "OHE",
                        OneHotEncoder(
                            sparse_output=True, handle_unknown="infrequent_if_exist"
                        ),
                        categorical_cols,
                    ),
                    # handle categorical columns as one hot encoding
                    (
                        "Text",
                        CountVectorizer(analyzer=stemmed_words),
                        "text",
                    ),
                    *[
                        (
                            col,
                            TfidfVectorizer(
                                ngram_range=(1, 4),
                                preprocessor=lambda x: " ".join(x.split(";")),
                                analyzer="word",
                            ),
                            col,
                        )
                        for col in [
                            "member_of_indirect_groups_ids",
                            "member_of_groups_ids",
                        ]
                    ],  # handle categories as text, interaction up to 4 included
                ]
            ),
            VarianceThreshold(threshold=0.001),
            PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            ),  # intersting interaction features like Is_admin * title
            VarianceThreshold(),  # remove zero varience features
            StandardScaler(
                with_mean=False
            ),  # Scaling is important for anomaly detection
        ),
    ),
    (
        "Features TfidfVectorizer with Stemmer",
        make_pipeline(
            ColumnTransformer(
                [
                    (
                        "OHE",
                        OneHotEncoder(
                            sparse_output=True, handle_unknown="infrequent_if_exist"
                        ),
                        categorical_cols,
                    ),
                    # handle categorical columns as one hot encoding
                    (
                        "Text",
                        TfidfVectorizer(analyzer=stemmed_words),
                        "text",
                    ),
                    *[
                        (
                            col,
                            TfidfVectorizer(
                                ngram_range=(1, 4),
                                preprocessor=lambda x: " ".join(x.split(";")),
                                analyzer="word",
                            ),
                            col,
                        )
                        for col in [
                            "member_of_indirect_groups_ids",
                            "member_of_groups_ids",
                        ]
                    ],  # handle categories as text, interaction up to 4 included
                ]
            ),
            VarianceThreshold(threshold=0.001),
            PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            ),  # intersting interaction features like Is_admin * title
            VarianceThreshold(),  # remove zero varience features
            StandardScaler(
                with_mean=False
            ),  # Scaling is important for anomaly detection
        ),
    ),
]
# Define algorithms:
anomaly_algorithms = [
    #     ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
    (
        "Isolation Forest",
        IsolationForest(contamination=outliers_fraction, random_state=42),
    ),
    # (
    #     "Local Outlier Factor",
    #     LocalOutlierFactor(n_neighbors=10, contamination=outliers_fraction),
    # ),
]

rng = np.random.RandomState(42)
for i_pipe, (pipe_name, pipeline) in enumerate(transform_algorithms):
    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        detector = make_pipeline(pipeline, algorithm)
        detector.fit(data)
        detector
        t1 = time.time()
        print(
            "Elapsed time for pipeline {} and algorithm {}: {:.2f}s".format(
                pipe_name, name, t1 - t0
            )
        )
        y_pred = detector.predict(data)
        print(f"Number of outliers: {sum(y_pred == 0)}")
print("No new outliers with ML")
anomalies = pd.concat([anomalies_1, anomalies_2], axis=0).drop(1616, axis=0)
anomalies.to_csv("anomalies.csv")
