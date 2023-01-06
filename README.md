# DisasterResponse

## Objective
This project analyzes real messages that were sent during disaster events. The objective is to create a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

## Getting started
To clone this project with SSH key authentication type:
```
git@github.com:Thomas-lj/DisasterResponse.git project_name
```

This project was run using Python 3.10. To setup this project, create a virtual environment, and install the project requirements:
```
python3 -m venv venv
pip install -r requirements.txt
```

## Project files
This is the folder setup for the project

    .
    ├── ...
    ├── app                                 # website layout design
    │   ├── template
            ├── master.html                 # main page of web app
            └── go.html                     # classification result page of web app
        ├── run.py                          # Getting started guide

    ├── data                            	# data and data processing folder
        ├── disaster_categories.csv         # data to process
        ├── disaster_messages.csv           # data to process
        ├── disasterdatab.db                # database to save clean data to
        └── process_data.py                 # data processing script

    ├── models                            	# data and data processing folder
        ├── train_classifier.py             # model classifier
        ├── classifier.pkl                  # saved model 

    README.md
    ...


## Run project
To execute the Disaster Response, first run the process_data.py ETL pipeline script that cleans data and stores in database.

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

Now you can train your ML pipeline that trains classifier and save the model in a .pkl file. Note that you can adjust the hyperparameters by changing the parameters variable.

```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

Now you can run your own homepage by executing this:
```
cd app
python run.py
```

Run locally that opens a localhost http://172.24.34.237:3001/ where you can view the dataset distribution and classify a new message of your own definition

![disaster_start_page](https://user-images.githubusercontent.com/43189719/211047472-b4410ab3-517f-4dc0-a7df-76e4bbeefcfa.png)
