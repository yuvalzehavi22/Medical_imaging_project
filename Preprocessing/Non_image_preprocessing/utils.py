import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def convert_categorical_to_numeric(df_given):
    df = df_given.copy()
    #  hold category-to-code mappings
    category_mappings = {}
    exclude_columns = ['filename','note','gpt4_summary','use']

    # Idientify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]

    # convert each categorical column to numeric
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column])

        # get category-to-code mapping
        mapping = dict(enumerate(df[column].cat.categories))
        category_mappings[column] = mapping

        #keep NaN values
        df[column] = df[column].cat.codes.replace(-1, np.nan)

    return df, category_mappings


#--------- sort the feaures in the order of the images to match - according to image_id ---------
# Function to sort features based on image IDs
def sort_features_by_image_ids(features, desired_idxs):
    # # Convert image_ids to a DataFrame for merging
    # image_id_order = pd.DataFrame({'image_id': image_ids})
    # # Merge features with image_id_order based on 'image_id' to align the rows
    # sorted_features = pd.merge(image_id_order, features, on='image_id', how='left')
    # Reindex the features DataFrame based on the provided desired_idxs
    sorted_features = features.reindex(desired_idxs).reset_index(drop=True)
    return sorted_features



def preprocess_cat_numeric(train_set):
    # Define feature types
    categorical_features = ['gender', 'race', 'ethnicity', 'language', 'maritalstatus']
    continuous_features = ['age']

    # Preprocessing for categorical data
    # categorical_transformer = Pipeline(steps=[
    #     ('onehot', OneHotEncoder(handle_unknown='ignore'))
    # ])

    # Fit scaler on training data and then use it to transform both training and test data (ensures that the test data is scaled based on the training data's statistics, maintaining consistency.)
    scaler = StandardScaler()
    scaler.fit(train_set[continuous_features])  # Fit on training data
    continuous_transformer = Pipeline(steps=[
        ('scaler', scaler)
    ])

    # # Combine transformers into a preprocessor
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         # ('cat', categorical_transformer, categorical_features),
    #         ('cont', continuous_transformer, continuous_features)
    #     ]
    # )
    # Combine transformers into a preprocessor, with categorical features first
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', 'passthrough', categorical_features),  # Pass categorical features as is
            ('cont', continuous_transformer, continuous_features)  # Scale continuous features
        ]
    )

    # Apply preprocessing to both train and test sets
    X_train_processed = preprocessor.fit_transform(train_set)

    # X_test_processed = preprocessor.transform(test_set)

    # categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

    return X_train_processed #, X_test_processed#, list(categorical_feature_names)