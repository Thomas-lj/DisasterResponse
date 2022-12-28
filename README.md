# DisasterResponse

## Objective
This project analyzes disaster response social media posts and builds an
This project analyzes real messages that were sent during disaster events. The objective is to create a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

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
To execute the Disaster Response
`python run.py`
