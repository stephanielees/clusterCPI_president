# clusterCPI_president
time series clustering for Consumer Price Index (CPI) series that has been sliced based on the US presidents

## Datasets
`usa_inflation_series_bls.xlsx` is the CPI data. The data is obtained from [The Bureau of Labor Statistics](bls.gov). The series starts from January 1913 to January 2024, but I segmented it using the USA presidents' term. Thus, we have 19 time series in our dataset. This dataset has unequal lengths of series since the duration of each presidency may vary.

The dataset `us-presidents.csv` is used to get the starting term of the presidents.

The video for `USAinflation_tslearn` is [here](https://youtu.be/WcnTyKP55Z8). This video talks about using agglomerative clustering for clustering CPI. The input for a clustering algorithm is a distance matrix, and the details of it is discussed in [here](https://youtu.be/OAS6ttB7u-Y). The code for that video is `ddtw_tslearn`.
