import pathlib
import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np

class CPaths:
    pathroot = pathlib.Path().resolve()
    GivenContentFolder = "GivenContent"
    DataFolder = os.path.join(GivenContentFolder , "Data")
    OpportunityData =  os.path.join(DataFolder , "SalesForce_Opportunity.csv")

    @staticmethod
    def PathTo(const):
        return os.path.abspath(os.path.join(CPaths.pathroot ,const))

    @staticmethod
    def GetPathToGivenContent():
        return CPaths.PathTo(CPaths.GivenContentFolder)

    @staticmethod
    def GetPathToDataFolder():
        return CPaths.PathTo(CPaths.DataFolder)

    @staticmethod
    def GetPathToOpportunityData():
        return CPaths.PathTo(CPaths.OpportunityData)



class GetData:

    def getOpportunityData(self):
        df= pd.read_csv(CPaths.PathTo(CPaths.OpportunityData))
        df = df.drop("Id" , axis=1)
        bools = ["IsDeleted" , "IsPrivate" , "StageSortOrder" , "IsClosed" , "IsWon"  , "HasOpportunityLineItem" , "FiscalYear"]
        for c in bools:
            df[c] = df[c].astype("category")

        for c in df.columns:
            if c.lower().find("date") > -1:
                if df[c].isna().sum() == df.shape[0]:
                    df = df.drop([c] , axis=1)
                else:
                    df[c] = pd.to_datetime( df[c] ,infer_datetime_format=True )
            elif len(df[c].unique()) < 10 or c.find("ID") > 0:
                df[c] = df[c].astype("category")
            elif df[c].dtype != np.number:
                df[c] = df[c].astype("category")

        # column /data type corrections go here
        return df



def getNumericColumns(df, exceptCols):
    return [c for c in df.columns if ptypes.is_numeric_dtype(df[c]) and c not in exceptCols]

def getCategoricalColumns(df):
    return [c for c in df.columns if ptypes.is_categorical_dtype(df[c])]
