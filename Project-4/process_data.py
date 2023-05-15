import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Cleans the dataset

    Parameters:
    df: azdias or population dataset

    Returns:
    df: a cleaned dataset

    """

    print("Dropping null columns above 0.5")
    null_columns = df.columns[df.isnull().mean() > 0.5]
    df = df.drop(columns=null_columns)


    #CAMEO_DEU_2015
    print("CAMEO_DEU_2015")
    #print(df["CAMEO_DEU_2015"].unique())
    df.loc[df['CAMEO_DEU_2015'] == 'XX', "CAMEO_DEU_2015"] = np.NaN

    print("\tfactorizing")
    #factorizing the object columns
    df["CAMEO_DEU_2015_factorized"] = pd.factorize(df['CAMEO_DEU_2015'])[0]


    #CAMEO_DEUG_2015
    print("\nCAMEO_DEUG_2015: replacing `X` with np.NaN")
    #replace 'X' with -1
    df.loc[df['CAMEO_DEUG_2015'] == 'X', "CAMEO_DEUG_2015"] = np.NaN
    #convert to float, int can not handle NaN
    print("\tConverting to float")
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].astype(float)


    #CAMEO_INTL_2015
    print("\nCAMEO_INTL_2015: replacing `XX` with np.NaN")
    df.loc[df['CAMEO_INTL_2015'] == 'XX', "CAMEO_INTL_2015"] = np.NaN
    print("\tConverting to float")
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].astype(float)


    print("\nFactorizing object columns")
    print("\tD19_LETZTER_KAUF_BRANCHE")
    df["D19_LETZTER_KAUF_BRANCHE_factorized"] = pd.factorize(df['D19_LETZTER_KAUF_BRANCHE'])[0]
    print("\tOST_WEST_KZ")
    df["OST_WEST_KZ_factorized"] = pd.factorize(df['OST_WEST_KZ'])[0]

    print("\nCreating column`EINGEFUEGT_AM_ordinal` and changing it to datetime ")
    #convert EINGEFUEGT_AM to datetime
    df["EINGEFUEGT_AM"] = pd.to_datetime(df['EINGEFUEGT_AM'])
    print("\tChanging `EINGEFUEGT_AM_ordinal` to ordinal")
    #create column _ordinal
    df["EINGEFUEGT_AM_ordinal"] = df.loc[df["EINGEFUEGT_AM"].notnull()]["EINGEFUEGT_AM"].apply(lambda x: x.toordinal())
    print("\tChanging `EINGEFUEGT_AM_ordinal` to to float")
    df["EINGEFUEGT_AM_ordinal"] = df['EINGEFUEGT_AM_ordinal'].astype(float)

    
    return df