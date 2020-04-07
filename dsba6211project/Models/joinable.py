from dsba6211project.Utils.getData import CPaths, GetData

data = CPaths.GetPathToDataFolder()
odf = GetData().getOpportunityData()

import os
import pandas as pd
import numpy as np

for f in os.listdir(data):
    if f.find(".csv") > -1:
        path = os.path.abspath(os.path.join(data , "Campaign.csv"))
        df = pd.read_csv(path ,  encoding = "ISO-8859-1", error_bad_lines=False)
        c = [c for c in df.columns if c.lower().find("campaign") > -1]
        if len(c) > 0:
            c = c[0]
            print(f)
            mdf = df.merge(odf , left_on=c , right_on="CampaignId" , how="inner")
            print(mdf.shape)


path = os.path.abspath(os.path.join(data , "Campaign.csv"))
df = pd.read_csv(path ,  encoding = "ISO-8859-1", error_bad_lines=False)

df.head()

df.shape
df.isna().sum()
tooMuchMissing = df.columns[np.where(df.isna().sum() > 0.8*df.shape[0] )]

mostlyZeros = df.columns[np.where(df[df == 0].count() > 0.8*df.shape[0] )]

df = df.drop(list(tooMuchMissing)  + list(mostlyZeros), axis=1)

df.shape
df.head()
df.columns

df["ExpectedRevenue"].describe()
df[["ExpectedRevenue" , "ParentId"]].sort_values("ExpectedRevenue" , ascending=False)
df["ExpectedRevenue"].quantile(0.90)


df["ExpectedRevenueScaled"] =df["ExpectedRevenue"].clip(upper=df["ExpectedRevenue"].quantile(0.75))

from plotnine import *

pltdf = df[["ExpectedRevenueScaled" ,"Id" ,"ParentId"]].dropna()
ggplot(pltdf , aes(x="Id" , y="ParentId" , fill="ExpectedRevenueScaled")) + geom_tile()


pltdf = df[["ExpectedRevenueScaled" ,"Type" ,"ParentId"]].dropna()
(ggplot(pltdf , aes(x="Type" , y="ParentId" , fill="ExpectedRevenueScaled")) +
    geom_tile() +
    theme(axis_text_x = element_text(angle = 90, hjust = 1)) )


def getState(val):
    if val == None or val == "nan":
        return None
    else:
        print(val)
        l = str(val).split(",")
        if len(l) == 1:
            return l[0]
        else:
            return l[1].strip(" ")

df["Location_State"] = df["Program_Location__c"].apply(getState)
df["Location_State"].value_counts()

from numpy import nansum
from numpy import nanmean

df["ExpectedRevenue"].sum()
df[df["Location_State"] == "nan"]["ExpectedRevenue"].sum()

df[["Location_State" , "ExpectedRevenue"]].groupby("Location_State").agg(expected_revenue_mean = ("ExpectedRevenue"  ,lambda x: x.mean(skipna=True)))

pltdf = df["Location_State"].value_counts().rename_axis('location').reset_index(name='counts')
pltdf = pltdf[pltdf["location"] != "nan"]
(ggplot(pltdf , aes(x="location" , y="counts")) +
    geom_col() +
    coord_flip()+
    theme(axis_text_x = element_text(angle = 90, hjust = 1)) )



df["CampaignLength"] = pd.to_datetime(df[ "EndDate"])- pd.to_datetime(df[ "StartDate"])
df["CampaignLength"] = df["CampaignLength"].dt.days.astype("float")
pltdf = df["CampaignLength"].value_counts().rename_axis("length").reset_index(name="counts")

ggplot(df , aes(x="CampaignLength" , y="ExpectedRevenueScaled" , color="Type")) + geom_point() + theme(legend_position="top")


df["HierarchyBudgetedCost"].value_counts()

pltdf = df["Workshop_Type__c"].value_counts().rename_axis("workshop_type").reset_index(name="count")

ggplot(pltdf , aes(x="workshop_type" , y="count" )) + geom_col() + theme(axis_text_x = element_text(angle = 65, hjust = 1))


pltdf = df["Workshop_Sponsor__c"].value_counts().rename_axis("Workshop_Sponsor__c").reset_index(name="count")

ggplot(pltdf , aes(x="Workshop_Sponsor__c" , y="count" )) + geom_col() + theme(axis_text_x = element_text(angle = 65, hjust = 1))



pltdf = df[["Workshop_Type__c" ,"Workshop_Sponsor__c" ,"ExpectedRevenueScaled"]].dropna(how="all")

pltdf = pltdf[ pltdf["ExpectedRevenueScaled"] < 5000]
(ggplot(pltdf , aes(x="Workshop_Sponsor__c" , y="Workshop_Type__c" , fill="ExpectedRevenueScaled")) +
    geom_tile() +
    theme(axis_text_x = element_text(angle = 65, hjust = 1)) )
    
