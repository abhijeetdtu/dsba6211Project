%load_ext autoreload
%autoreload 2

from sklearn.ensemble import RandomForestRegressor


from dsba6211project.Utils.getData import *


df = GetData().getOpportunityData()


df.isna().sum()

df.shape


def drop_too_many_missing(df , fraction=0.7):
    counts =df.isna().sum()
    return df.drop(counts[counts > df.shape[0]*fraction].index , axis=1)

def split_date_cols(df):
    cols = []
    for c in df.columns:
        if c.lower().find("date") > -1:
            cols.append(c)
            #df[f"{c}_year"] , df[f"{c}_month"] = df[c].dt.year , df[c].dt.month
            #df[f"{c}_year"]  = df[f"{c}_year"].astype("category")
            df[f"{c}_day"] = df[c].dt.day
            df[f"{c}_day"]  = df[f"{c}_day"].astype("category")
            df[f"{c}_month"] = df[c].dt.month
            df[f"{c}_month"]  = df[f"{c}_month"].astype("category")

    df = df.drop(cols , axis=1)
    return df

def drop_id_cols(df):
    return df.drop(["AccountId" , "RecordTypeId"] , axis=1)
#tooManyMissingCols = tooManyMissing(X)

df = df.pipe(drop_too_many_missing , fraction=0.7) \
    .pipe(split_date_cols) \
    .pipe(drop_id_cols)

df["Amount"].describe()



cat_cols =  getCategoricalColumns(df)
num_cols =  getNumericColumns(df , exceptCols=["Amount" , "stayclassy__sc_order_id__c"])
cols_useless= ["ExpectedRevenue", "npe01__Payments_Made__c","CampaignId", "OwnerId", "CreatedById", "Tax_Letter__c" , "stayclassy__Campaign__c","LastModifiedById", "SystemModstamp","npsp__Primary_Contact__c" ,"npe01__Contact_Id_for_Role__c", "stayclassy__sc_order_id__c","stayclassy__sf_contact_id__c"]
cats_to_dummy = [c for c in cat_cols if c not in cols_useless]

def impute(df , cat_cols , num_cols):
    df.loc[:,cat_cols] = df.loc[:,cat_cols].fillna(df.loc[:,cat_cols].mode().iloc[0])
    df.loc[:,num_cols] = df.loc[:,num_cols].fillna(df.loc[:,num_cols].mean())
    return df

def get_dummies(df , cat_cols):
    return pd.get_dummies(df , columns=cat_cols , drop_first=True)

tdf = df.pipe(impute , cat_cols , num_cols)
    #.pipe(get_dummies , cats_to_dummy)

tdf = tdf[~tdf["Amount"].isna()]


from sklearn.preprocessing import KBinsDiscretizer

tdf["Amount"] = KBinsDiscretizer(n_bins=5, encode='ordinal').fit_transform(tdf[["Amount"]])[:,0]

#tdf["Amount"].describe()

X = tdf.drop(["Amount"] + cols_useless ,axis=1)
y = tdf["Amount"]


from statsmodels.api import MNLogit

model = MNLogit(X,y)
results = model.fit()

print(results.summary())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

accs = cross_val_score(RandomForestClassifier(), X,y, cv=5 , scoring="accuracy")
accs.mean()

from sklearn.model_selection import train_test_split

X_train, X_test , y_train , y_test = train_test_split(X,y)

reg = RandomForestClassifier()
reg.fit(X_train,y_train)


plotdf = pd.DataFrame({"cols" : X.columns , "imp" : reg.feature_importances_})


plotdf.sort_values("imp" , ascending=False).head(30)

ggplot(plotdf , aes(x="cols" , y="imp")) + geom_col()
