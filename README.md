# TestDocker

How to run

1. `git clone https://github.com/StefanPushkov/TestDocker.git`

2. `cd TestDocker`

3. `docker build -t stef/test_project .`


4. `docker run -p 8888:8888 stef/test_project`


### Submission csv file: `./submission_voting_classifier.csv`

Graphics: 
-  `./aucNN.jpg` - roc curve for neural network prediction
-  `./aucClassisML.jpg` - roc curve for voting classifier (classic ML)

