# Wildfire Cause Detector
-----------------------

The US experiences an average of over 70,000 wildfires each year, burning an estimated 7.0million acres.# This widespread destruction, which has been consistently growing in magnitude each year, is a mounting public safety concern where 4.5million US homes have been identified to be at high or extreme risk from wildfires.# Aside from the enormous environmental and disaster relief issues this presents, it also incurs significant economical repercussions where $5.1billion in losses have accrued over the last 10 years due to wildfires.#

In order to reduce the devastating impact presented by these fires, it is important to be able to determine the underlying causes. By quantitatively understanding the different wildfire antecedents, we can help target efforts to dramatically curb the spread of wildfires. This can manifest, for instance, through educational strategies, redirection of resources, concerted mobilization tactics, etc. 

To this end, the framework presented in this repository attempts to construct a deep learning neural network to determine whether a wildfire was caused by natural or human-based sources. The data used to train the model was assembled from three separate sources that include the data for all US wildfires from 1992-2015, as well as historical climate and stock data covering the same time period (links are provided in the Installation description below). The motivation behind use of the climate data was to offer a link between wildfire frequency and associated temperatures & temperature fluctuations in the region. To complement the environmental data, historical stock information was included to be a potential indicator for economically driven factors; for example, dramatic drops in the market might lead to more crime (e.g., arson in this case), or might motivate people to take up less expensive hobbies (e.g., a possible statistical increase in camping and therefore an increase in campfires). Furthermore, smoking behavior has been loosely tied to economic downturns, and serves as another nontrivial culprit behind wildfires.#

Training and subsequent hyperparameter tuning of the neural network led to a respectable 87% accuracy for prediction of natural/human causes on the test set of fires between 2011 and 2014 - outperforming previous models by a ~15% margin. As such the model can be helpful in assigning the causes behind US wildfires, especially in ambiguous or unknown cases, hopefully improving our understanding of wildfire genesis and assisting with further prevention. 


Files and Repository Structure
-----------------------

### Directory Tree

Add directory tree. 

### File Descriptions

Here are brief descriptions of the files made available in this repository (refer to above tree for file locations):
* `main.py` is the ‘main’ Python script to initiate processing of the data, feature engineering, pipeline creation, training and testing, etc.
* `settings.py` lists the user defined variables for running main.py and the listed packages.
* `requirements.txt` list all the libraries needed to execute the code as well as the version of the library, best for use in a virtual environment.
* `data_info.txt` provides a copy of the links to the different datasets and specifies that the data files should be located in the data directory. 
* `collate_data.py` processes the individual data sets, engineers relevant features, and combines that data into one SQL database (stored as combined_data.db after execution).
* `separate_time_data.py` loads the data from the new SQL database, assigns relevant datatypes to features, and separates the data into training, validation, and test sets by time series.
* `prepare_pipeline.py` constructs a pipeline to process the numerical, categorical, and cyclical time features before training.
* `convert_to_cycle.py` contains the custom transformer for cyclical features (i.e., days, months, years, etc.) by applying a sine and cosine transformation. 
* `deep_neural_network_classifier.py` contains a wrapper for the Tensorflow construction of a multicalss deep neural network that in addition to defining number of hidden layers and neurons, supports dropout, batch normalization, and early stopping.
* `dnn_wildfires.*` are a series of files that contain the trained DNN model.
* `measure_performance.py` includes a helper function for measuring model performance on a trained neural network.

Installation
-----------------------

### Download the data

* Clone this reposoitory to your computer.
* Get into the data folder using `cd data`.
* Download the data files for wildfire, climate, and stock histories into the `data` directory.  
    * You can find the US wildfire data [here](https://www.kaggle.com/rtatman/188-million-us-wildfires).
    * You can find the historical climate data [here](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data#GlobalLandTemperaturesByCity.csv).
        * The only file needed is `GlobalLandTemperaturesByCity.csv`.
    * You can find the historical stock data [here](https://www.kaggle.com/ehallmar/daily-historical-stock-prices-1970-2018).
        * The only file needed is `historical_stock_prices.csv`.
* Extract the files you downloaded.
    * Place `FPA_FOD_20170508.sqlite` directly into the data directory.
    * Place `GlobalLandTemperaturesByCity.csv` directly into the data directory.
    * Place `historical_stock_prices.csv` directly into the data directory.
* Switch back into the `wildfire-cause-detector` directory using `cd ..`.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt` or `pip3 install -r requirements.txt`, depending on your system.
    * Make sure you are using Python 3.
    * You may want to use a virtual environment to run the program.

Usage
-----------------------

* Move to the `wildfire-cause-detector` directory.
* Run `python main.py` or `python main.py` (depending on system) to process & combine datasets, perform feature engineering, train & test model.
    * This will create `combined_data.db` in the `data_collated` folder.
    * This will also create `dnn_wildfires.*` in the `models` folder.

References
-----------------------

https://fas.org/sgp/crs/misc/IF10244.pdf

https://www.verisk.com/insurance/campaigns/location-fireline-state-risk-report/

https://www.ncbi.nlm.nih.gov/m/pubmed/23956058/

