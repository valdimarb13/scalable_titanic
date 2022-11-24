# scalable_titanic

Lab 1 for
ID2223 / HT2022

Predicting if passenger of the Titanic survive with a Serverless ML System.
The data cleaning pipeline is based on https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy#5.12-Tune-Model-with-Hyper-Parameters.

The feature pipeline cleans the data and registers the resulting dataset as a feature group on Hopsworks.

The training pipeline constructs a Hopsworks feature view and trains a Decision Tree classifier, the resulting model is also saved to Hopsworks.

Using the model from Hopsworks it's possible to see how a user created passenger might fare aboard the Titanic:
https://huggingface.co/spaces/Valdimarb13/titanic

The daily feature pipeline creates a synthetic passenger based on the known passengers by changing a subset of their features.

The batch inference pipeline predicts whether these synthetic passengers can survive, it's possible see the results of the most recent predictions here:
https://huggingface.co/spaces/Valdimarb13/titanic-monitor


