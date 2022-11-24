import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("titanic_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_person(person):
    """
    Given a real passenger as input gives a synthetic passenger with one random, randomized feature
    """
    import pandas as pd
    import random
    #Changing one of the features
    possible_features = ['sex_code','pclass', 'embarked_code', 'title_code', 'familysize', 'agebin_code', 'farebin_code']
    randomnumber = random.randint(0,6)
    generator_list = [random.randint(0, 1), random.randint(1,3), random.randint(0, 2), random.randint(0, 4), random.choice([1,2,3,4,5,6,7,8,11]), random.randint(0, 4), random.randint(0, 3)]
    person[possible_features[randomnumber]] = generator_list[randomnumber]
    return person


def get_synthetic_passenger(fs):
    """
    Returns a DataFrame containing one random  synthetic passenger
    """
    import pandas as pd
    import hopsworks

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    query = titanic_fg.select_all()
    dataset = query.read(online=False, dataframe_type="default", read_options={})
    synthetic_passenger = generate_person(dataset.sample())

    return synthetic_passenger


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    possible_features = ['sex_code','pclass', 'embarked_code', 'title_code', 'familysize', 'agebin_code', 'farebin_code']

    titanic_df = get_synthetic_passenger(fs)
  
    titanic_fg = fs.get_or_create_feature_group(
    name="titanic_synthetic_passenger",
    version=1,
    primary_key = possible_features,
    description="Titanic")
    print(titanic_df)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic_daily")
        with stub.run():
            f()
