import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    titanic_fg = fs.get_feature_group(name="titanic_synthetic_passenger", version=1)
    query = titanic_fg.select_all()
    dataset = query.read(online=False, dataframe_type="default", read_options={})

    labels = list(dataset['survived'])
    batch_data = dataset.drop(columns=['survived'])

    last_entry = batch_data.iloc[-1:]
    label = labels[-1]

    #These two lines need to be changed above and below
    survival = model.predict(last_entry)[0]

    if(survival==1):
        survival_url = "https://images.pexels.com/photos/8384595/pexels-photo-8384595.jpeg?auto=compress&cs=tinysrgb&w=600"
    else:
        survival_url = "https://images.pexels.com/photos/1121906/pexels-photo-1121906.jpeg?auto=compress&cs=tinysrgb&w=600"
    print("Predicted: " + str(survival))
    img = Image.open(requests.get(survival_url, stream=True).raw)            
    img.save("./latest_prediction.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_prediction.png", "Resources/images", overwrite=True)
   
    
    if(label==1):
        label_url = "https://images.pexels.com/photos/8384595/pexels-photo-8384595.jpeg?auto=compress&cs=tinysrgb&w=600"
    else:
        label_url = "https://images.pexels.com/photos/1121906/pexels-photo-1121906.jpeg?auto=compress&cs=tinysrgb&w=600"
    print("Actual value: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_label.png")
    dataset_api.upload("./actual_label.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Survival Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survival],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our prediction feature group has examples of survival and non survival
    print("Number of different survival predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['Did not survive', 'Survived'],
                        ['Predicted not to survive', 'Predicted to survive'])

        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different survival predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different survival predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

