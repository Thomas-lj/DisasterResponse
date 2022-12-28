# DisasterResponse

## Objective
This project analyzes disaster response social media posts and builds an

## Project files
This is the folder setup for the project
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
    |   classifier.pkl           # saved model 
    └──
    - README

or, less commonly, into the `doc` folder.

    .
    ├── ...
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── TOC.md              # Table of contents
    │   ├── faq.md              # Frequently asked questions
    │   ├── misc.md             # Miscellaneous information
    │   ├── usage.md            # Getting started guide
    │   └── ...                 # etc.
    └── ...


## Run project
To execute the Disaster Response
`python run.py`
