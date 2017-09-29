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
    return year

def get_month(date):
    month = date.split('-')[1]
    return month

def get_day(date):
    day = date.split('-')[2]
    return day

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
    print(df[:3])
    plot_corr_mat(df)


