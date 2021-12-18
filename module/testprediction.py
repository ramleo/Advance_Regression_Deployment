def pred_test(data=None, trans_miss=None, trans_onehot_scale=None, rfe_vars=None, model=None):
    '''Predicting the test dataset'''

    # Importing relevant packages
    from module.datacleaner import leads_data_cleaner
    from module.createvariables import create_variables
    from module.ordinalinverse import ord_inv_transforming
    from module.encoding import encode
    import pandas as pd
    import warnings

    # Cleaning the data
    data = leads_data_cleaner(data=data)

    # Creating variables
    num_miss, cat_miss, onehot_vars, total_vars, List = create_variables()

    # create a for loop to data into numeric
    for col in cat_miss:
        warnings.filterwarnings("ignore")
        encode(data=data[col])

    # Transforming the dataset
    df = pd.DataFrame(trans_miss.transform(data))

    # Renaming the coulmns
    df.columns = total_vars

    # Performing inverse transform
    df = ord_inv_transforming(data=df)

    # Creating dataframe after transformation
    df = pd.DataFrame(trans_onehot_scale.transform(df), columns=List)

    # Selecting top 10 features for test dataset
    df_test = df[rfe_vars]

    # Predict test dataset
    y_pred = model.predict(df_test)

    return df_test, y_pred