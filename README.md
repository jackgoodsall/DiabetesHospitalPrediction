# Diabetes problem

This is my repo for an end-to-end machine learning project using mlflow for predicting readmission of patients from 100k+ records of medical data.

This project designs to implement and design a fully functional pipeline with reuseable and decouled components for data ingestion, data engineering and model training and evaluation.

Uses logging and mlflow for debugging and experiment tracking respectively. 


This project is not about getting the highest performance for the specific task and more about implementing and using best MLOPs practices, and in theory this project can be easily changed for any other tasks by modifing the correct internal components for that specific task (the scaling, imputers, feature pipelines, models, metrics, etc).

## How to use

To run model training pipeline define what models you want to train inside the config and the parameters with which you wish to train them. 


## Docker image

This project can be run in a docker image using the contained docker container - you just need to build it.
