FROM python:3.8

RUN mkdir /app
WORKDIR /app

RUN apt update -y
RUN pip install --upgrade pip
RUN python -m pip install tensorflow
RUN python -m pip install numpy
RUN python -m pip install opencv-python-headless
RUN python -m pip install psycopg2
RUN python -m pip install nltk
RUN python -m pip install pandas
RUN python -m pip install scikit-learn

#CMD python index.py
#CMD python dictionary.py
CMD python classification.py