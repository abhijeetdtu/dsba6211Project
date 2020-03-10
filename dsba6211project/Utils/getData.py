import pathlib
import os
import pandas as pd

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
        # column /data type corrections go here
        return df


df = GetData().getOpportunityData()

df.shape
