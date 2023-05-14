import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifcats","model.pkl")
            preprocessor_path = os.path.join('artifcats','preprocessor.pkl')
            print("before loading the model")
            model = load_object(file_path=model_path)
            print("after loading the model")
            preprocessor = load_object(file_path=preprocessor_path)
            print("before transforming")
            data_scaled = preprocessor.transform(features)
            print("after transforming")
            preds = model.predict(data_scaled)
            print("after prediction")

            return preds

        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    def __init__(  self,
        batting_team : str,
        bowling_team : str,
        city : str,
        current_score : int,
        wickets_left: int,
        last_five : int,
        balls_left : int,
        crr : int, 
        ):

        self.batting_team = batting_team

        self.bowling_team = bowling_team

        self.city = city

        self.current_score = current_score

        self.wickets_left = wickets_left

        self.last_five = last_five

        self.balls_left = balls_left
        
        self.crr = crr 


    def get_data_as_data_frame(self):
        try:
            # balls_left = 120 - (self.over*6)
            # last_five = self.current_score/self.over
            custom_data_input_dict = {
                "batting_team": [self.batting_team],
                "bowling_team": [self.bowling_team],
                "city": [self.city],
                "current_score": [self.current_score][0],
                "balls_left": [self.balls_left],
                "wickets_left": [self.wickets_left][0],
                "crr": [self.crr],
                "last_five": [self.last_five][0],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)