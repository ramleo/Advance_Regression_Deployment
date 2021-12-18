def impute_scale_trans(num_vars=None, ord_vars=None, onehot_vars=None, total_vars=None, list_ord_vars=None, list1=None):
    '''Creating column transformer for missing data and categorical variables'''

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from module.datacleaner import data_cleaning
    from sklearn.preprocessing import (FunctionTransformer,
                                        OneHotEncoder, StandardScaler, OrdinalEncoder)

    # Imputing missing values by using 'median' for numeric and 'most_frequent' for categorical
    ct_missing = ColumnTransformer(transformers=[
        ('num_impute', SimpleImputer(strategy='median'), ['LotFrontage', 'MasVnrArea']),
        ('cat_impute', SimpleImputer(strategy='most_frequent'), ['MasVnrType', 'Electrical'])], remainder='passthrough')

    # Onehot encoding and ordinal encoding the variables
    ct_onehot_ord = ColumnTransformer(transformers=[
        ('one_hot', OneHotEncoder(sparse=False,handle_unknown='ignore'), onehot_vars),
        ('ord', OrdinalEncoder(categories=list_ord_vars), ord_vars)],
        remainder='passthrough')

    # Standard scaling the variables
    ct_scale = ColumnTransformer(transformers=[('scale', StandardScaler(), ord_vars + num_vars)],
                                 remainder='passthrough')

    # Using 'FunctionTransformer' to use udf in the 'Pipeline' function
    ft1 = FunctionTransformer(data_cleaning)

    # Creating a function to convert numpy array after imputation into a dataframe
    def conv_df_missing(data):
        df = pd.DataFrame(data, columns=total_vars)
        return df

    # Using 'FunctionTransformer' to use udf in the 'Pipeline' function
    ft2 = FunctionTransformer(conv_df_missing)

    # Creating a function to convert numpy array after onehot and ordinal encoding into a dataframe
    def conv_df_onehot(data):
        df = pd.DataFrame(data, columns=list1 + ord_vars + num_vars)
        return df

    # Using 'FunctionTransformer' to use udf in the 'Pipeline' function
    ft3 = FunctionTransformer(conv_df_onehot)

    # Creating a function to convert numpy array after standard scaling into a dataframe
    def conv_df_scale(data):
        df = pd.DataFrame(data, columns=ord_vars + num_vars + list1)
        return df

    # Using 'FunctionTransformer' to use udf in the 'Pipeline' function
    ft4 = FunctionTransformer(conv_df_scale)

    return ct_missing, ct_onehot_ord, ct_scale, ft1, ft2, ft3, ft4