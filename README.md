## What is the purpose of this project?
The goal of this project was to create forecast system for GoogleAdWords data. GoogleAdWords data consists of the following records:
{date,impressions,clicks,conversions,cost,total_conversion_value,average_position,reservations,price}</br>
In this project i created two seperate models for predicting click & conversion values for the next day.</br></br>
## Click values forecast
Mean absolute error values:</br>
SVM score: 3.7(polynomial kernel with degree of 2)</br>
KNN score: 4.3(k=11)</br>
NN score: 5.5</br> </br>
## Conversions values forecast
Mean absolute error values:</br>
SVM score: 0.27(linear kernel)</br>
KNN score: 0.35(k=5)</br>
NN score: 0.27</br> </br>
## Some interesting plots:
![alt text](https://github.com/PiotrSobczak/GoogleAdWords-Forecast-Tensorflow-Sklearn/blob/master/plots/next_clicks_clicks.png)
![alt text](https://github.com/PiotrSobczak/GoogleAdWords-Forecast-Tensorflow-Sklearn/blob/master/plots/next_clicks_day_of_month.png)
![alt text](https://github.com/PiotrSobczak/GoogleAdWords-Forecast-Tensorflow-Sklearn/blob/master/plots/next_clicks_impressions.png)
![alt text](https://github.com/PiotrSobczak/GoogleAdWords-Forecast-Tensorflow-Sklearn/blob/master/plots/next_clicks_movavg.png)
