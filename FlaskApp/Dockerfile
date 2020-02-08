FROM python:3-onbuild

COPY . /FlaskApp
# We copy just the requirements.txt first to leverage Docker cache
#ADD requirements.txt /

WORKDIR /FlaskApp

#RUN pip install -r requirements.txt
RUN pip3 install Flask
RUN pip3 install sklearn
RUN pip3 install scikit-learn
RUN pip3 install scipy
RUN pip3 install smart-open
RUN pip3 install tqdm
RUN pip3 install ConfigParser
RUN pip3 install pandas
RUN pip3 install numpy

EXPOSE 5000

#ENTRYPOINT [ "python" ]

CMD ["python", "./app.py" ]