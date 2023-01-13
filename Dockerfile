FROM python:3.11-bullseye
RUN pip install scikit-learn matplotlib pandas seaborn numpy nltk
# Need to pip freeze and testing working
# RUN pip install jupyter black #for analytics ipynb.
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt
COPY ueba.csv ueba.csv
COPY anomaly_detection.py anomaly_detection.py
RUN pip freeze > requirements.txt
# ENTRYPOINT ["python", "anomaly_detection.py"]
