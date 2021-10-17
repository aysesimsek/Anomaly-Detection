# Detecting Anomalies in Smart Buildings using Machine Learning
## **Designs for data collection:**
Python scripts were created to automate data collection and processing. These scripts use python's built-in json module to extract data from json files and the request module to collect json files from the Urban Observatory REST API. In this way, the REST services were queried with the script for a summary for all rooms in the observatory. Next, the summary JSON file was used to gather the room entity IDs. At this point, it was important to choose more frequently used rooms to detect an abnormality in a sensor data. With the occupancy sensor, this frequency of use can be determined. For this reason, services were queried for more detailed information using the room entity IDs obtained, and the return results were filtered according to the rooms with occupancy sensors. Occupancy information was requested to find the top 5 rooms with the highest traffic in a selected month (2019-08). 
Using this information, all sensor occupancy information was collected for 4 rooms, sensor IDs and metric measured by the sensor were stored, and 3 rooms were selected according to the number of similar measurements in these rooms. Then, depending on the API's response, the rest service was queried for timeseries data for rooms using sensor IDs that collect data every two years, monthly or daily. Finally, the timeseries json files collected were merged into CSV files in the format of Figure 1.

 |![image](https://user-images.githubusercontent.com/37701256/137637222-74c75ad6-fec3-472a-a4ea-bcb5e60c9373.png)|
 | :-: |
 Figure 1

Figure 1. This figure shows the layout of the CVS file that is produced when all the sensors for one room are merged.

The combined CVS files were then read into the python pipelines for the models, ready for feature engineering and pre-processing. But first some sensors needed to be dropped, we found the best sensors to run the models on would be CO2, Relative Humidity, Room Brightness, Room Temperature. All the other were removed, either because they were binary, such as room occupancy, or ambiguous as heating set point and mode, these were:  

Actual Cooling Set point, Actual Heating Set point, Chilled Water Valve, CO2, Cooling Set Point, Cooling Valve Position, Heating Set Point, Heating Valve Position, HVAC Operating Mode, Light Power Level, Low Temperature Hot Water Valve, Mode, Mode Input, Occupancy Sensor, Relative Humidity, Room Brightness, Room Occupied, Room Temperature, BrightnessValueZ2, Fan Speed.

The rooms used for the project were: 2.008, G.069, 6.031.
## **Selected models:**
- K-means
- Isolation Forest:
## **Pre-processing data:**
**K-means:**

The features used in k-means were time and value, no other factor was considered in the training and the testing of the models, because the classifier is based on cluster formation these features made the most sense. Metric and duration were the key features that were not taken into consideration directly in the data set used. Two new columns were created in the data, P1 and P2, P1 being Value and P2 being time. Time represents time of day and did not include the date the data was taken on, such that all data points fit with in a 24-hour window. Because of this all values were scaled together using the standard scalar. 

Value however was scaled separately per metric, because the range of the values and the breadth of the values will be different per metric, so too are all the data points. Scaling per metric means that all the datapoints can be in one large cluster, this is advantageous because the other option is to have all sensors in one dataset but to separate them, which makes the clustering algorithm struggle. In this format using the most optimum number of clusters, the algorithm places clusters between two sensors in empty space, which will cause anomalies to not appear. So scaling them such that they can all overlap in one larger cluster is advantages.


 |![image](https://user-images.githubusercontent.com/37701256/137637235-97c04216-6540-4f0e-9fd9-8f10fac77067.png)|
 | :-: |
 Figure 2

Figure 2 above shows the test set with all sensors overlapped and grouped by the model. This test data is for room 2.008. 

**Isolation forest:**

The datasets represent multivariate timeseries, we chose to build a generalised model that is flexible and has low bias so that the model is adaptable and performs better with increasing complexity.  For the generalised model to work well within the context of the data available, the Metric and Duration (error window) features were reintroduced. The metric feature was encoded using a label encoder, this is due to only one categorical feature being encoded. The encoder labels classes between 0 and classes -1, for our chosen metric this turned out to encode values from 0 to 3. Scaling was not needed as it has little to no effect on the Isolation Forest. 

Rolling windows(means) needed to be calculated to ensure the stability of our model over time. Timeseries models trained on features excluding the rolling means struggle to detect contextual anomalies. Rolling windows in a timeseries work by calculating the means of the target feature for a given time window e.g., 1 hour, 1 day, 1 month, etc... This helps forecast what values can be expected in the next window based on results of the previous window. For example, if the previous rolling window forecast a concave upwards, and given a set of values for the next window has a few values that concave downwards and deviate from the other values, they should identify as contextual anomalies. For our model, smaller rolling windows measured at 1 hour, 30 minutes, and 45 minutes for the value, then 20 minutes, 15 minutes, and 10 minutes for the duration. These values were chosen based on fitting the rolling window for a given day, the rolling windows that did not overfit or underfit the variance of values were chosen.




 |![image](https://user-images.githubusercontent.com/37701256/137637248-dad1dc18-76dc-4cd3-b7aa-d2c33078d982.png)|
 | :-: |
 Figure 3

 Figure 3. shows how reactive the rolling mean is to the variance of values when calculated at 1-hour, 6-hour and 12-hour intervals. 


 |![image](https://user-images.githubusercontent.com/37701256/137637251-19a9d50d-f175-43b1-b504-1aa865a54ae6.png)|
 | :-: |
 Figure 4
 
 Figure 4. shows how the rolling window measured at each 1h fits to the variance of values for 3 given days. See appendix H for other rolling window graphs.


## **Hyper parameter training:**
**K-means:**

To do the Hyper-parameter training a loop was used for k-means, because all the hyper-parameters are numeric. The loop can be used to run through a range of values and after each iteration the score function can be used to find the most optimum value for that given hyper-parameter. After the loop has been completed a graph is produced, called an elbow curve, that displays the results. 



 |![image](https://user-images.githubusercontent.com/37701256/137637258-ae2ec6ab-47b3-4b8d-9223-a744d0c4f118.png)|
 | :-: |
 Figure 5

The results for the hyper parameter training were:

- ‘n\_clusters’ : 6
- ‘n\_init’ : 20
- ‘max\_iter’ : 20

The other graphs for the hyper parameters can be seen in appendix B.

**Isolation forest:**

All 3 datasets were labelled with predicted anomalies using a biased Isolation forest model with its contamination set to 1%, max\_features set to 2 and n\_estimators set to 100. This is because the contamination sets the limit for the number of outliers in the dataset. The labelled anomalies will be used as true values to test our model against. To improve the perform of our model, the hyperparameters max\_samples, max\_features and n\_estimators were put in GridSearchCV to find the best possible combination of hyperparameters for each dataset. The contamination value is left to default to prevent it from affecting our model’s forecasting. After splitting each dataset between training and test sets, the training set was used in the GridSearch with its labelled anomalies as the target outputs. The GridSearch tested the best hyperparameters on f1 scoring. The results of GridSearch on the training sets with hyperparameters to use on our models shows that it detects between 2 – 3 % of anomalous data for all 3 datasets.



 |![image](https://user-images.githubusercontent.com/37701256/137637262-b78a6e43-d5ec-46cf-b2dc-06baf86b4f7a.png)|
 | :-: |
 Figure 6

## **Results after Training the algorithms:**
**K-means:**

K-means inherently does not have the capabilities to detect anomalies, and so a function was defined. The function first finds the distance of all datapoints relative to their respective cluster centroids. This the length of the list is then multiplied by the outlier fraction, which is determined by the programmer, to find how many outliers there may be in the data. Using this value an array of the largest distance values is found. The threshold value is the minimum of this array. Finally, if the distance from the data points centroid is greater than or equal to the threshold value than it is considered an anomaly.


 |![image](https://user-images.githubusercontent.com/37701256/137637267-28b9c39c-97da-49be-b109-3572bb039130.png)|
 | :-: |
 Figure 7

Figure 7 is the test data from room 2.008, these are the results after running the anomaly function on the dataset after it has been grouped, figure 2. The red dots represent anomalies. Other rooms data is in appendix A.



 |![image](https://user-images.githubusercontent.com/37701256/137637275-eb451f42-6326-4a2c-b160-cc54154ad15a.png)|
 | :-: |
 Figure 8

Figure 8 shows CO2 data for room 2.008 represented over the 2-month period, against the value. The red X’s show where the above anomalies detected in figure 3 are respective to time and value.

The other sensors result like in figure 4 can be found in appendix C, this also includes Isolation forests data.

**Isolation Forest:**

Isolation Forest is a very good algorithm for detecting anomalies, what the algorithm simply does is that it divides the data into two parts based on the threshold value recursively until each data point is isolated. Data points which take fewer steps will be considered as outliers. After using the GridSearch function to find the best values for hyperparameters, the algorithm detects about 2%-3% anomalies.



 |![image](https://user-images.githubusercontent.com/37701256/137637285-e968c7c7-e8d1-473c-bae5-dc5afd37c869.png)|
 | :-: |
 Figure 9

Figure 9 shows the results after running the algorithm on the dataset of room 2.008 against DateTime, Duration and Value. The red dots in the figure represent anomalies it detects. 



 |![image](https://user-images.githubusercontent.com/37701256/137637290-123800f3-5830-45c9-92cc-29d9a3fa0709.png)|
 | :-: |
 Figure 10

Figure 10 illustrates the CO2 data set of room 2.008 over a two-month period. The red dots in the figure show the above detected outliers are respective to time and value. 


## **The design of experiments:**
### Evaluation:
**Design:** 

The destine of the evaluation experiment was to compare the detection capabilities of both algorithms. This is especially hard because the data itself does not have a validation column, and so visualisations of the data will be used to compare the two algorithms.

The experiment itself is simply to take all the data sets in the three datasets and predict all the data according to the room's respective models. Here we can find where the algorithms agree on a given data point, and which data they disagree on. 

**Results:**

Table 1
|**For all data**|**K-mean anomalies**|**K-mean non-anomalies**|
| :- | :- | :- |
|Isolation forest anomalies|393|3208|
|Isolation forest non-anomalies|777|112867|


Table 1 lumps all 3 datasets into one and divides the data points into 1 of 4 categories, where they agree, if k-means and Isolation forest both predict that a datapoint is anomalies or not. And where they disagree, where one model or the other predicts a point to be anomalies and the other does not. 

 |![image](https://user-images.githubusercontent.com/37701256/137637295-91938e93-b78b-4f20-bc15-cb3572381398.png)|
 | :-: |
 Figure 11

Figure 11 shows data from Room 2.008 for CO2 with both the models predicted anomalies, the red x represents K-means anomalies and the yellow X Isolation forests. 

**Evaluation:**

Overall Isolation forest found more anomalies, as can be seen in figure 7, yellow dots are scattered across the all the data, this will be in part due to the fact that Isolation forest was using 4 features rather than 2 like k-means, which were value, time, duration, and metric whereas k-means was only using Value and time. Because of this k-means only finds anomalies at the peaks and troughs of the data, see appendix D. additionally what can be seen is the two algorithms agreeing at the extreme points of the graphs. The extra features used in Isolation Forest seem to have made it produce false positives, anomalies are scattered across the graph for Isolation forest, this can only be confirmed visually.
## Injection attack:
**Design:** 

Injection attack was performed by taking an hour data from each dataset and increasing and decreasing a given data point’s value till their respective model sated it as an anomaly. This was repeated for two more times on different hours and days for all datasets on both algorithms.

What this produce is the upper and lower values for each sensor of each room for the given 3 hours.

To do this a subset of the data is taken from the hour on the given day on that month, and a new data value was synthetically made, whose value is the average value for the given sensor being tested. This value is then increased, re-scaled and re predicted till the prediction is an anomaly. This process was done for each sensor, on each data set in each hour chosen.

**Results:**

Table 2
|**K-Means September 11th 1pm For room 2.008**|**Upper (Anomalous Values)**|**Lower (Anomalous Values)**|**Upper (Deviation from the mean)**|**Lower (Deviation from the mean)**|
| :- | :- | :- |:- |:- |
|Room Temperature|25|17|+3|-4|
|CO2|577|299|+105|-173|
|Room Brightness|1136|-236|+747|-623|
|Relative humidity|71|18|+21|-31|



Table 3
|**Isolation forest September 11th 1pm For room 2.008**|**Upper (Anomalous Values)**|**Lower (Anomalous Values)**|**Upper (Deviation from the mean)**|**Lower (Deviation from the mean)**|
| :- | :- | :- |:- |:- |
|Room Temperature|23|18|+2|-3|
|CO2|534|<p>383</p><p></p>|<p>+62</p><p></p>|-89|
|Room Brightness|462|462|N/A|N/A|
|Relative humidity|60|33|+11|-16|

Table 2 and 3 show the upper and lower bounds of both types of classifiers for room 2.008 in September the 11th at 1pm. All the other results can be seen in appendix G.

## **Evaluation:**

Isolation forest works on a Rolling Mean, some results do not have an upper or lower bound as all results will return an anomaly, such is the case of room brightness on September 11th 1pm. After running these tests, Isolation forest produces a smaller allowance of deviation from the mean value before considering a datapoint anomalous. This is supported by the combined graphs in appendix D. Additionally the injection attack for Isolation forest also seems to predict much faster than k-means, and there for would be better in a situation of finding anomalies in a stream of data. 

## **Future improvements:** 
The first improvement would-be real-world testing by Incorporating the trained models into a program that can be ran while real data is being streamed from the sensors. We determined a time range and compared the algorithm results over the sensor data in this range. This time interval can be diversified, and algorithms can be tried over more rooms. According to the results, hyperparameters can be made even more efficient. Because of trying to detect anomaly with unsupervised algorithms, we cannot know whether the sensor data labelled as anomalous is anomalous. At this point, comparing the results of algorithms applied to sensor data in other time intervals can provide us with an idea of the accuracy of the data labelled as anomalous. Again, the results can be observed by plotting results on graphs which helps in evaluating the models but having confirmed anomalous data would help in determining the accuracy of them. Testing the algorithm on sensor data for more than 2 months may provide more realistic results.







