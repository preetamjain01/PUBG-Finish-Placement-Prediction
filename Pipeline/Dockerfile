FROM python:3-onbuild

COPY . /FlaskPipeline

WORKDIR /FlaskPipeline

RUN pip3 install Flask
RUN pip3 install sklearn
RUN pip3 install scikit-learn
RUN pip3 install scipy
RUN pip3 install smart-open
RUN pip3 install tqdm
RUN pip3 install ConfigParser
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install luigi
RUN pip3 install seaborn
RUN pip3 install matplotlib


EXPOSE 5000

CMD ["python", "./Pipeline.py" ]