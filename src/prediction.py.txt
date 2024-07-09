import numpy as np

import logging

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



logger = logging.getLogger()



def prepare_data(temperature, humidity, wind_speed):

    """Prepares data for training the prediction model."""

    X = np.stack((temperature, humidity, wind_speed), axis=-1)

    X = X.reshape(-1, 3)

    return X



def train_model(X, y, model_type='linear'):

    """Trains a machine learning model for weather prediction with hyperparameter tuning."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    if model_type == 'linear':

        model = Pipeline([

            ('scaler', StandardScaler()),

            ('regressor', LinearRegression())

        ])

    elif model_type == 'random_forest':

        model = Pipeline([

            ('scaler', StandardScaler()),

            ('regressor', RandomForestRegressor())

        ])

        param_grid = {

            'regressor__n_estimators': [50, 100, 200],

            'regressor__max_depth': [None, 10, 20, 30]

        }

        model = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

    else:

        raise ValueError("Unsupported model type. Choose 'linear' or 'random_forest'.")

    

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    logger.info(f"Model {model_type} training score: {score:.2f}")

    

    return model



def predict_future(model, current_data):

    """Predicts future weather conditions using the trained model."""

    return model.predict(current_data)

