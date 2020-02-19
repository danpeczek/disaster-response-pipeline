# disaster-response-pipeline

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

- To install all required packages and run tests you need run from root directory:
    - Install dependencies from requirements.txt:
        - `pip install -r requirements.txt`
    - Make modules visible locally:
        - `pip install .`
    - Run tests:
        - `pytest`

## Project Motivation <a name="motivation"></a>

This project is a part of the Udacity Data Science Nano Degree program - 
https://www.udacity.com/course/data-scientist-nanodegree--nd025.

The goal of this project is to demonstrate usability of classification model in context 
of disaster information. The project runs a web-application that can classify message 
about disaster in 36 categories. Usage of this classifier can make easier decision where
particular message should be sent to make a good action e.g. when message contains
information about food or medicines are required, then this message can be sent to 
parties connected supplying food/medicines.

## File Descriptions <a name="files"></a>
* data/process_data.py - provides ETL pipeline that cleans and loads clean data to database 
based on csv files containing messages about disasters. Dataset is provided by courtesy of
FigureEight (https://www.figure-eight.com/).
* models/train_classifier.py - provides Machine Learning pipeline that takes database 
generated with script above and trains classifier and tune hyper-parameters.
* app/run.py - runs web-service that presents diagrams based on the generated database and 
allows to classify custom user messages.

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
Database and classifier name should remain the same, as currently webapp does not distinguish 
different names.
    - To run ETL pipeline that cleans data and stores in database
        - `python data/process_data.py disaster_messages.csv disaster_categories.csv messages.db`
    - To run ML pipeline that trains classifier and saves classifier
        - `python models/train_classifier.py messages.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    - `python app/run.py`

3. Go to http://0.0.0.0:3001/

Important note: if you want only to see how classifier works and you don't have csv 
files that were used for training, **but** you have my classifier then just use operation from point 3.
You will not have visualisation on the overview of the dataset, but you'll still be able to classify custom 
message.  

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

The data for classifier were available by courtesy of FigureEight (https://www.figure-eight.com/),
who provided the data for the Udacity Data Science Nanodegree course (https://www.udacity.com/course/data-scientist-nanodegree--nd025).