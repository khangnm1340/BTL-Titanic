This script performs data loading, preprocessing, feature engineering, model training, and prediction for the Titanic survival prediction problem.

## Part 1: Setup and Initial Data Loading

* **Import Libraries**: Imports necessary libraries for data manipulation, visualization, and machine learning.
* **Create Output Directory**: Checks if a directory named "titanic\_images" exists, and creates it if not, to save generated plots.
* **Load Data**: Loads `train.csv` and `test.csv` into pandas DataFrames.

## Part 2: Exploratory Data Analysis and Feature Engineering

* **Display Basic Information**: Shows the head, info, and descriptive statistics of the training DataFrame, including null value counts.
* **Analyze Survival by Categorical Features**:
    * Calculates the mean survival rate grouped by `Pclass`, `Sex`, `SibSp`, and `Parch`.
* **Feature Engineering: Family Size**:
    * Creates a `Family_Size` feature by summing `SibSp` (siblings/spouses) and `Parch` (parents/children) and adding 1 (for the passenger themselves).
    * Maps `Family_Size` to `Family_Size_Group` (Alone, Small, Medium, Large) for better categorization.
    * Analyzes survival rate by `Family_Size_Group`.
    * **Plot**: Generates and saves a histogram of `Family_Size` distribution.
    * **Screenshot 1: `family_size_distribution.png` should be here.**
* **Analyze Survival by Age**:
    * **Plot**: Generates and saves a distribution plot of `Age` by `Survived`.
    * **Screenshot 2: `age_distribution_by_survival.png` should be here.**
    * **Feature Engineering: Age Binning**:
        * Creates `Age_Cut` by quantiling the `Age` feature into 8 bins.
        * Binning: Replaces continuous `Age` values with integer categories (0-8) based on these quantile cuts.
        * **Plot**: Generates and saves a histogram of original `Age` with quantile cut points.
        * **Screenshot 3: `original_age_with_cuts_distribution.png` should be here.**
        * **Plot**: Generates and saves a histogram of the binned `Age` categories.
        * **Screenshot 4: `binned_age_distribution.png` should be here.**
* **Analyze Survival by Fare**:
    * **Plot**: Generates and saves a distribution plot of `Fare` by `Survived`.
    * **Screenshot 5: `fare_distribution_by_survival.png` should be here.**
    * **Feature Engineering: Fare Binning**:
        * Creates `Fare_Cut` by quantiling the `Fare` feature into 8 bins.
        * Binning: Replaces continuous `Fare` values with integer categories (0-8) based on these quantile cuts.
        * **Plot**: Generates and saves a histogram of original `Fare` with quantile cut points.
        * **Screenshot 6: `original_fare_with_cuts_distribution.png` should be here.**
        * **Plot**: Generates and saves a histogram of the binned `Fare` categories.
        * **Screenshot 7: `binned_fare_distribution.png` should be here.**
* **Feature Engineering: Title from Name**:
    * Extracts `Title` (e.g., Mr., Mrs., Miss) from the `Name` column.
    * Normalizes various titles into broader categories (e.g., "Mlle" to "Miss", "Capt" to "Military").
    * Analyzes survival rate by `Title`.
* **Feature Engineering: Name Length**:
    * Calculates `Name_Length`.
    * **Plot**: Generates and saves KDE plots of `Name_Length` for survived and not-survived passengers.
    * **Screenshot 8: `Name_length_Survived.png` should be here.**
    * Binning: Divides `Name_Length` into `Name_Size` categories (0-8) based on quantiles.
    * **Plot**: Generates and saves a histogram of `Name_Length` distribution.
    * **Screenshot 9: `name_length_distribution.png` should be here.**
* **Feature Engineering: Ticket Information**:
    * Extracts `TicketNumber` (the last part of the ticket string).
    * Calculates `TicketNumberCounts` (how many times a ticket number appears).
    * Extracts `TicketLocation` (the prefix of the ticket string if present).
    * Normalizes various `TicketLocation` values.
* **Feature Engineering: Cabin Information**:
    * Fills missing `Cabin` values with "U" (unknown).
    * Extracts the first letter of the `Cabin` as the new `Cabin` category.
    * Creates `Cabin_Assigned` (0 if 'U', 1 otherwise).
    * Analyzes survival rate by `Cabin` and `Cabin_Assigned`.
* **Handle Missing Numerical Values**:
    * Fills missing `Age` and `Fare` values with their respective means.

## Part 3: Model Preprocessing Pipeline

* **Define Preprocessing Steps**:
    * `OrdinalEncoder`: For features like `Family_Size_Group`.
    * `OneHotEncoder`: For categorical features like `Sex` and `Embarked`.
    * `SimpleImputer`: To handle missing values (though most are handled, this acts as a safeguard).
* **Define Target and Features**:
    * `X`: Features for training (dropping `Survived`).
    * `y`: Target variable (`Survived`).
    * `X_test`: Features for prediction.
* **Split Data**: Divides the training data into `X_train`, `X_valid`, `y_train`, `y_valid` using `train_test_split` with stratification.
* **Create Preprocessing Pipelines**:
    * `ordinal_pipeline`: Imputes missing values with the most frequent, then applies `OrdinalEncoder`.
    * `ohe_pipeline`: Imputes missing values with the most frequent, then applies `OneHotEncoder`.
* **ColumnTransformer**:
    * Applies different preprocessing steps to different columns:
        * `Age` is imputed.
        * `Family_Size_Group` uses `ordinal_pipeline`.
        * `Sex` and `Embarked` use `ohe_pipeline`.
        * `Pclass`, `TicketNumberCounts`, `Cabin_Assigned`, `Name_Size` are passed through without transformation.

## Part 4: Model Training and Evaluation (with GridSearchCV)

* This section systematically trains and tunes several classification models using `GridSearchCV` for hyperparameter optimization and `StratifiedKFold` for robust cross-validation.
* **RandomForestClassifier (rfc)**:
    * Defines a parameter grid for `n_estimators`, `min_samples_split`, `max_depth`, `min_samples_leaf`, and `criterion`.
    * Performs `GridSearchCV`.
    * Prints the best parameters and best cross-validation score.
    * **Screenshot 10: Print output for `CV_rfc.best_params_` and `CV_rfc.best_score_` should be here.**
* **DecisionTreeClassifier (dtc)**:
    * Defines a parameter grid.
    * Performs `GridSearchCV`.
    * Prints the best parameters and best cross-validation score.
    * **Screenshot 11: Print output for `CV_dtc.best_params_` and `CV_dtc.best_score_` should be here.**
* **KNeighborsClassifier (knn)**:
    * Defines a parameter grid.
    * Performs `GridSearchCV`.
    * Prints the best parameters and best cross-validation score.
    * **Screenshot 12: Print output for `CV_knn.best_params_` and `CV_knn.best_score_` should be here.**
* **SVC (svc)**:
    * Defines a parameter grid.
    * Performs `GridSearchCV`.
    * Prints the best parameters and best cross-validation score.
    * **Screenshot 13: Print output for `CV_svc.best_params_` and `CV_svc.best_score_` should be here.**
* **LogisticRegression (lr)**:
    * Defines a parameter grid.
    * Performs `GridSearchCV`.
    * Prints the best parameters and best cross-validation score.
    * **Screenshot 14: Print output for `CV_lr.best_params_` and `CV_lr.best_score_` should be here.**
* **GaussianNB (gnb)**:
    * Defines a parameter grid.
    * Performs `GridSearchCV`.
    * Prints the best parameters and best cross-validation score.
    * **Screenshot 15: Print output for `CV_gnb.best_params_` and `CV_gnb.best_score_` should be here.**

## Part 5: Prediction and Output

* **Make Predictions**: Uses the `pipefinaldtc` (Decision Tree Classifier with the best parameters from GridSearchCV) to make predictions on the `X_test` dataset.
* **Save Results**:
    * Creates a new DataFrame `test_results_df` by copying `test_df`.
    * Adds the `Survived_Prediction` column to `test_results_df`.
    * Reorders columns for better readability.
    * Saves the `test_results_df` to a CSV file named `detailed_predictions.csv`.
    * Reads the saved CSV back in for verification.
    * Prints the head of a subset DataFrame containing `PassengerId` and `Survived_Prediction`.
    * **Screenshot 16: The head of the `subset_df` (PassengerId and Survived_Prediction) printed to console should be here.**
