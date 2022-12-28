# DisasterResponse

## Objective
This project seeks to analyze disaster response social media posts and predict them appropriately.

## Project files
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

.
├── ...
├── app                    # Test files (alternatively `spec` or `tests`)
│   ├── template              
  │   ├── master.html         # main page of web app
  │   └── go.html             # classification result page of web app
    run.py                    # Flask file that runs app
└── 
├── data
│   disaster_categories.csv  # data to process 
│   disaster_messages.csv    # data to process
│   disaster_messages.csv    # data to process
│   process_data.py          # scripts that cleans and saves data.
└── ...
├── models
│   train_classifier.py      # model classifier
|   classifier.pkl  # saved model 
└──
- README
