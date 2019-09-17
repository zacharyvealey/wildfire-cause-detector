import time
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV

from settings import Settings
import data_collated
import models
import training
import testing


def run_pipeline():
    """Initiate and run machine learning analysis on US wildfires."""
    # Initialize settings
    start_time = time.time()
    ml_settings = Settings()
    
    # Collate data from different data sets (i.e., US wildfires, climate, 
    # and stocks), engineer features, and merge into a single SQL database.
    data_collated.collate_data(ml_settings)

    # Separate data into training, validation, and test sets (by dividing up)
    # time series then shuffling.
    X_train, X_val, X_test, y_train, y_val, y_test = training.separate_time_data(ml_settings)

    # Engineer a pipeline to impute and standardize where appropriate as well
    # as make certain time features cyclical and include one hot encoding
    # where necessary.
    prep_pipeline = training.prep_pipeline(ml_settings)

    X_train_prepared = prep_pipeline.fit_transform(X_train)
    X_val_prepared = prep_pipeline.transform(X_val)

    # Create a deep learning neural network framework then train the model
    # on the data using early stopping.
    print('Training Deep Neural Network:')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    dnn_clf = models.DNNClassifier(random_state=42, 
                                n_hidden_layers=ml_settings.n_hidden_layers, 
                                n_neurons=ml_settings.n_neurons, 
                                batch_size=ml_settings.batch_size,
                                batch_norm_momentum=ml_settings.batch_norm_momentum, 
                                dropout_rate=ml_settings.dropout_rate, 
                                learning_rate=ml_settings.learning_rate,
                                activation=ml_settings.activation)

    dnn_clf.fit(X_train_prepared, y_train, n_epochs=1000, X_valid=X_val_prepared, y_valid=y_val)

    # Perform a randomized hyperparameter search using a time series cross validation.
    if ml_settings.hp_search:
        # Define time series cross validation and parameter distribution.
        ts_cv = TimeSeriesSplit(n_splits=3).split(X_train_prepared)
        param_distribs = {
            "n_neurons": [50, 100, 200, 300],
            "batch_size": [50, 100, 500],
            "learning_rate": [0.01, 0.02, 0.05, 0.1],
            "activation": [tf.nn.elu, tf.nn.leaky_relu],
            "n_hidden_layers": [1, 2, 3, 4, 5],
            "batch_norm_momentum": [0.9, 0.95, 0.99, 0.999],
            "dropout_rate": [0.2, 0.4, 0.6],
        }

        # Run randomized hyperparameter search.
        rnd_search = RandomizedSearchCV(models.DNNClassifier(random_state=42), param_distribs, n_iter=50,
                                        cv=ts_cv, random_state=42, verbose=1, n_jobs=-1)
        rnd_search.fit(X_train_prepared, y_train, n_epochs=1000, X_valid=X_val_prepared, y_valid=y_val)
        print(rnd_search.best_params_, '\n')

    # Measure trained model on test set.
    X_test_prepared = prep_pipeline.transform(X_test)

    if ml_settings.hp_search:
        testing.measure_performance(X_test_prepared, y_test, rnd_search, 
                    show_classification_report=True, show_confusion_matrix=True)
    
    else:
        testing.measure_performance(X_test_prepared, y_test, dnn_clf, 
                    show_classification_report=True, show_confusion_matrix=True)

    # Save trained neural network.
    if ml_settings.hp_search:
        rnd_search.best_estimator_.save(ml_settings.MODEL_PATH)
    else:
        dnn_clf.save(ml_settings.MODEL_PATH)

    # Report computational time.
    end_time = time.time()
    total_time = end_time - start_time
    print('Time: ', total_time)
    
if __name__ == "__main__":
    run_pipeline()