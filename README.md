# GoogleAdWords Click-Conversion Forecast
Creating forecast system for GoogleAdWords data. GoogleAdWords data consists of the following records: </br>
{date,impressions,clicks,conversions,cost,total_conversion_value,average_position,reservations,price}</br>
The goal is to create 2 models for predicting click & conversion values for the next day.</br></br>
<b>Clicks value forecast</b></br>
Mean absolute error values:</br>
SVM score: 3.7(polynomial kernel with degree of 2)</br>
KNN score: 4.3(k=11)</br>
NN score: 5.5</br> 
<b>Converions value forecast</b></br>
Mean absolute error values:</br>
SVM score: 0.27(linear kernel)</br>
KNN score: 0.35(k=5)</br>
NN score: 0.27</br> 



