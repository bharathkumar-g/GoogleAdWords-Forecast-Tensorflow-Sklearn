import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

def plot_corr_mat(df):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    sns.set(font_scale=0.8)

    plt.show()

def get_year(date):
    year= date.split('-')[0]
    return int(year)

def get_month(date):
    month = date.split('-')[1]
    return int(month)

def get_day(date):
    day = date.split('-')[2]
    return int(day)

if __name__=='__main__':
    #Read data
    df = pd.read_csv('ad_data.csv')

    #Parsing date
    df['year'] = df.date.apply(get_year)
    df['month'] = df.date.apply(get_month)
    df['day'] = df.date.apply(get_day)
    df = df.drop('date',1)

    #Moving year,month,day columns to front
    df = df[['year', 'month', 'day','impressions', 'clicks', 'conversions', 'cost', 'total_conversion_value', 'average_position', 'reservations', 'price']]

    #Renaming columns names
    df = df.rename(index=str, columns={"average_position": "avg_position", "total_conversion_value": "tot_conversion_val"})

    #Getting clicks & coversions columns as lists
    clicks = df['clicks'].tolist()
    conversions = df['conversions'].tolist()

    #Moving each element up by removing first element. After that adding 0 to the end so that column size matched DF size
    clicks.pop(0)
    clicks.append(0)
    conversions.pop(0)
    conversions.append(0)

    #Creating 2 new columns with values of clicks&conversions of the next day
    df['next_clicks'] = clicks
    df['next_conversions'] = conversions

    #Plotting correlation matrix
    #plot_corr_mat(df)

    #Dropping last row, because we won't learn anything from it. We have already extracted the clicks and conversions.
    df = df[:-1]

    # Shuffling the data, keep the index
    df = df.sample(frac=1)

    #Dividing the set into input features and output,
    Y = df.ix[:,'next_clicks']
    X = df.drop(['next_clicks','next_conversions'],1)

    #Normalizing inputs
    X = (X - X.min() - (X.max() - X.min()) / 2) / ((X.max() - X.min()) / 2)
    
    print("X:",X)


