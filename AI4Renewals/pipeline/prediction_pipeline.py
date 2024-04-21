import os
import sys

import numpy as np
import pandas as pd
from AI4Renewals.entity.config_entity import AI4RenewPredictorConfig
from AI4Renewals.entity.s3_estimator import GenDesEstimator
from AI4Renewals.exception import AI4RenewException
from AI4Renewals.logger import logging
from AI4Renewals.utils.main_utils import read_yaml_file
from pandas import DataFrame


class GenDesData:
    def __init__(self,
                Country,
                user_education,
                Gender,
                LastLoginDaysAgo,
                Profession,
                SubscriptionType,
                UsesVR,
                Age,
                DesignProjectScale,
                NumberOfDesigns,
                TutorialProgressionType,
                FrequencyOfDesignToolUsage,
                CustomerSupportCall,
                AppEngagementMinutes,
                IsActive
                ):
        """
        GenDes Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Country = Country
            self.user_education = user_education
            self.Gender = Gender
            self.LastLoginDaysAgo = LastLoginDaysAgo
            self.Profession = Profession
            self.SubscriptionType = SubscriptionType
            self.UsesVR = UsesVR
            self.Age = Age
            self.DesignProjectScale = DesignProjectScale
            self.NumberOfDesigns = NumberOfDesigns
            self.TutorialProgressionType = TutorialProgressionType
            self.FrequencyOfDesignToolUsage = FrequencyOfDesignToolUsage
            self.CustomerSupportCall = CustomerSupportCall
            self.AppEngagementMinutes = AppEngagementMinutes
            self.IsActive = IsActive


        except Exception as e:
            raise AI4RenewException(e, sys) from e

    def get_gendes_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from GenDesData class input
        """
        try:
            
            gendes_input_dict = self.get_gendes_data_as_dict()
            return DataFrame(gendes_input_dict)
        
        except Exception as e:
            raise AI4RenewException(e, sys) from e


    def get_gendes_data_as_dict(self):
        """
        This function returns a dictionary from GenDesData class input 
        """
        logging.info("Entered get_gendes_data_as_dict method as GenDesData class")

        try:
            input_data = {
                "Country": [self.Country],
                "user_education": [self.user_education],
                "Gender": [self.Gender],
                "LastLoginDaysAgo": [self.LastLoginDaysAgo],
                "Profession": [self.Profession],
                "SubscriptionType": [self.SubscriptionType],
                "UsesVR": [self.UsesVR],
                "Age": [self.Age],
                "DesignProjectScale": [self.DesignProjectScale],
                "NumberOfDesigns": [self.NumberOfDesigns],
                "TutorialProgressionType": [self.TutorialProgressionType],
                "FrequencyOfDesignToolUsage": [self.FrequencyOfDesignToolUsage],
                "CustomerSupportCall": [self.CustomerSupportCall],
                "AppEngagementMinutes": [self.AppEngagementMinutes],
                "IsActive": [self.IsActive]
            }

            logging.info("Created gendes data dict")

            logging.info("Exited get_gendes_data_as_dict method as GenDesData class")

            return input_data

        except Exception as e:
            raise AI4RenewException(e, sys) from e

class GenDesClassifier:
    def __init__(self,prediction_pipeline_config: AI4RenewPredictorConfig = AI4RenewPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise AI4RenewException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of class GenDesClassifier:
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of GenDesClassifier class")
            model = GenDesEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise AI4RenewException(e, sys)