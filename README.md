# Overview
The objective of the project is to compare the performance of multiple machine learning models to evaluate the performance of each on a given dataset.  The dataset used is the [Airbnb Price Dataset]([https://renate.readthedocs.io/en/latest/index.html#cite-renate](https://www.kaggle.com/datasets/rupindersinghrana/airbnb-price-dataset/code)) found on Kaggle.  The dataset has various attributs which contribute towards its usefulness in a thorough ML project:

**Size:** The dataset has about 75,000 rows and 29 features before preprocessing.  Generally, more data is better when trying to build an ML model which can make accurate predictions.
**Features:** The features included in the dataset are of varying data types, including numerical, text, and binary values.  A variety of different features can contribute towards better predictions.
**Pre-processing potential:** While having clean data to begin with is convenient, it is still important to know how to work with unclean data.  Additionally, working through unclean data can lead towards a more thorough understanding of the data, since alterations need to be made with purpose.

As intended by the original dataset, the Log_price column (Airbnb price scaled by log base 10) was used as the target column, leading to a regression model.  The project was completed in a group of four people in accordance with the project requirements.  The contributions made by me specifically include the Decision Tree model and the SHAP diagram.

# Pre-Processing
Different features were pre-processed according to their data type.  Numerical features such as latitude and longitude were unchanged, while categorical features were transformed to numerical values with LabelEncoder from Scikitlearn.  Some features had "t" and "f" values which were mapped to 1 and 0 respectively.  The amenities feature, being a JSON string, required multiple steps.  Each column was turned into a Python list, filtered to remove unnecessary amenities, and one hot encoded with MultiLabelBinarizer.

StandardScaler was applied to numerical columns to reduce the effect of outliers skewing the data.  Some numerical features had missing values, which were imputed using a median strategy, also to avoid the effect of outliers. Finally, to promote uniformity, all feature names were changed to snake case, where words are separated with underscores rather than spaces.

# Exploratory Data Analysis
EDA was performed for various numerical features to observe data distributions.  Some key observations include:

-Target variable: A histogram of the target variable, log_price, shows an approximate normal distribution with a mean of about 4.5.
-Amenities: Histograms of amenities such as number of accomodates, number of bathrooms, etc. show common trends in how many guests the Airbnbs are designed to accomodate.  Most Airbnbs seem to accomodate 1-2 people, having 1 bedroom, 1 bathroom, etc.
-Correlation heatmap: Reveals highly correlated features.  It can be seen that accomodates, bedrooms, and beds have higher correlation with each other.

Further exploratory analysis can be viewed in the code file.

# Modeling Approach
As mentioned, the aim of the project was to compare multiple ML models to see which gave the best results.  This included hyperparameter tuning when possible to further maximize the accuracy of a model.  Except for the neural network which utilized Keras, all models utilize Scikitlearn.  The details of each approach are summarized below:

1) Neural Network: Sequential model with two hidden layers and two dropout layers, with number of nodes decreasing by half each layer.  Hyperparameter tuning is performed for the dropout layers, with rates (0.5, 0.3) giving a minimum MSE Of 0.4199.  Final model was trained for 100 epochs.
2) KNN: Hyp. tuning is performed for the value of k, with minimum MSE of 0.2359 for k=5.  KNeighborsRegressor model is fit with k=5 and evaluated.
3) Decision Tree: Hyp. tuning is performed with GridSearchCV for max_depth, min_samples_split, and max_features.  The best parameters come out as 12 for max_depth, 2 for min_samples_split, and 'sqrt' for max_features.  DecisionTreeRegressor is fit with the saved parameters from tuning, and evaluated.
4) Random Forest: RandomizedSearchCV is used for hyp. tuning to determine values for n_estimators, max_depth, min_samples_split, and min_samples_leaf.  Optimal parameter values come out to be 100, 20, 10, and 2, respectively.  The best parameters are saved, and used to fit a RandomForestRegressor model before evaluation.
5) Ridge Regression: GridSearchCV is used to tune the value of alpha, with the optimal value coming out to 10.  A Ridge model is fitted and evaluated using alpha=10.0.

# Results
Each model eas evaluated for Mean Squared Error (MSE), Mean Absolute Error (MAE), Root MSE, and R^2 value.  The results are summarized below:

| Model | Mean Squared Error | Mean Absolute Error | Root Mean Squared Error | R^2 |
| --- | --- | --- | --- | --- |
| Neural Network | 0.29 | 0.40 | 0.54 | 0.42 |
| KNN | 0.23 | 0.36 | 0.48 | 0.54 |
| Decision Tree | 0.23 | 0.35 | 0.48 | 0.54 |
| Random Forest | 0.16 | 0.28 | 0.40 | 0.69 |
| Ridge Regression | 0.23 | 0.37 | 0.48 | 0.53 |

From the results, we can confidently conclude that the Random Forest model performed the best on the dataset.  It obtained the lowest values for MSE, MAE, and RMSE, and the highest R^2 value.

In addition to the above metrics, a SHAP plot was generated for the Random Forest model.  A SHAP plot is a beeswarm plot which shows the features that had the highest effect on the model outputs.  The variance of the features is the key indicator of how impactful they were, as features showing a higher variance were more significant in determing the prediction given by the model.  For example, it is shown that the room_type feature has the highest impact on the output, with lower values contributing to a higher output, and vice versa.  The next most impactful features are latitude and longitude.  This makes sense, as Airbnbs in more densely populated areas are generally more expensive than ones in smaller cities.  On the other end, a feature such as host_identity_verified does not have much variance, and therefore was mostly insignificant in the predictions made by the model.  The full plot can be viewed in the code file.

# Conclusion
The project provides a thorough analysis of how different machine learning models perform on the given dataset.  The results are further supported by the inclusion of hyperparameter tuning for each model, ensuring that optimal parameters were used.  The Random Forest model performed the best among all available metrics, and features such as room_type and latitude/longitude had the highest effect on the model outputs.  A well trained model will be useful for both Airbnb customers and owners in their decisions for renting and pricing.
