# ADS-Final-Project-Prediction_On_PUBG_Dataset

## For running Flask on docker--

### Build-
docker build -t ankit08015/ads-flask-app-latest .

### Pull-
docker pull ankit08015/ads-flask-app-latest


### Run-

docker run -p {laptop port}:5000 ankit08015/ads-flask-app-latest

Now run on browser at {docker-ip}:{ip of laptop port given}
  
  
 ## For accessing the flask app via cloud click on following lin:-
 
 AWS ElasticBean Server- https://team5-ads-flask.herokuapp.com/
 
 
 ## For running Complete Pipeline on docker--

### Build-
docker build -t ankit08015/ads-final-pipeline .

### Pull-
docker pull ankit08015/ads-final-pipeline


### Run-

docker run -p ~/desktop/config.ini:/FinalPipeline/config.ini ankit08015/ads-final-pipeline
 
