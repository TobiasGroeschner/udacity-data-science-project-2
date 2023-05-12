import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

base_color = sb.color_palette()[0]
def creat_count_plot(plt_title:str, df:pd.DataFrame, plotted_column:str):
    """
    plt_title: Title of the plot
    df: DataFrame
    plotted_column: column to be plotted

    Plots a count plot with above arguments
    """
    plt.figure(figsize = [15, 5]) 
    
    plt.title(plt_title)
    sb.countplot(data = df, y = plotted_column, color = base_color,
             order = df[plotted_column].value_counts().index
             )

def data_exploration(df):
    print("SHAPE")
    print(df.shape)
    print("\n")

    print("NUMBER OF UNIQUES\n")
    print(df.nunique().sort_values(ascending= False).head(20))
    print("\n")
    
    print("NUMBER OF ABSOLUTE NULLS")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    print("\n")

    print("NUMBER OF % NULLS")
    print(df.isnull().mean().sort_values(ascending=False))
    print("\n")

    print("DUPLICATES")
    print("Numnber of duplicates in unique identifier: {}".format(len(df.loc[df["LNR"].duplicated()])))
    print("\n")

    print("Number of customers: {}".format(df.shape[0]))
    print("Number of features: {}".format(df.shape[1]))
    print("\n")

    print("VALUE COUNTS OF dtypes")
    print(df.dtypes.value_counts())
    print("\n")
