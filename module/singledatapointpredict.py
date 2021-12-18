def single_datapoint_predict(df=None, model=None, trans_miss=None, trans_onehot_scale=None):
    '''A function for predicting single data point'''

    # Importing relevant packages
    import pandas as pd
    import numpy as np
    from module.ordinalinverse import ord_inv_transforming
    from module.createvariables import create_variables
    from sklearn.preprocessing import OrdinalEncoder

    # Instantiating Ordinal encoder
    encoder = OrdinalEncoder()

    # Cleaning the data point
    df = df.drop('Prospect ID', axis=1)
    df.loc[df['Lead Source'] == 'google', 'Lead Source'] = 'Google'
    if df['Last Activity'][0] in ['Had a Phone Conversation', 'Approached upfront', 'View in browser link Clicked', \
                                  'Email Received', 'Email Marked Spam', 'Visited Booth in Tradeshow',
                                  'Resubscribed to emails']:
        df['Last Activity'].replace(df['Last Activity'], 'Others', inplace=True)
    if df['What is your current occupation'][0] in ['Housewife', 'Businessman', 'Other']:
        df['What is your current occupation'].replace(df['What is your current occupation'], 'Others', inplace=True)
    if df['Lead Source'][0] in ['Facebook', 'bing', 'Click2call', 'Live Chat', 'Social Media', 'Press_Release', \
                                'youtubechannel', 'NC_EDM', 'Pay per Click Ads', 'welearnblog_Home', 'WeLearn', \
                                'blog', 'testone']:
        df['Lead Source'].replace(df['Lead Source'], 'Others', inplace=True)
    if df['Lead Origin'][0] in ['Lead Import', 'Quick Add Form']:
        df['Lead Origin'].replace(df['Lead Origin'], 'Others', inplace=True)
    df.loc[df['Specialization'] == 'Select', 'Specialization'] = np.nan
    df.loc[df['How did you hear about X Education'] == 'Select', \
           'How did you hear about X Education'] = np.nan
    df.loc[df['City'] == 'Select', 'City'] = np.nan
    df.loc[df['How did you hear about X Education'].isnull(), 'How did you hear about X Education'] = 'Missing'
    df.loc[df['City'].isnull(), 'City'] = 'Missing'

    # Creating variables
    num_miss, cat_miss, onehot_vars, total_vars, List = create_variables()

    # Columns selected by sklearn RFE function
    rfe_vars = ['Lead Source_Olark Chat', 'Lead Source_Welingak Website', 'Last Activity_Converted to Lead',
                'Last Activity_Olark Chat Conversation', 'Last Activity_Others', 'Last Activity_SMS Sent',
                'What is your current occupation_Working Professional', 'Do Not Email_Yes', 'Lead Origin_Lead Add Form',
                'Total Time Spent on Website']

    # Removing column with 'Null Values'
    if df.isnull().sum().sum() > 0:
        df1 = df.dropna(axis=1)
    else:
        df1 = df

    # Creating list of variables with object type
    tmp_vars = []
    for i in df1.columns:
        if df1[i].dtype == 'O':
            tmp_vars.append(i)

    # Performing ordinal transform
    df1[tmp_vars] = encoder.fit_transform(df1[tmp_vars])

    # Saving feature with missing value to a variable
    miss = df.columns[np.squeeze(df.isna().values)]

    # Assigning values of transformed features to the dataframe
    df[df.columns[np.squeeze(~df.isna().values)]] = df1

    # Imputing missing values by performing transform ('SimpleImputer', 'IterativeImputer') and creating dataframe
    df = pd.DataFrame(trans_miss.transform(df), columns=total_vars)

    # Inverse transforming ordinal variables
    df[tmp_vars] = encoder.inverse_transform(df[tmp_vars])

    # Taking copy of the variables as 'ord_inv_transforming' function transforms dataframe inplace
    data = df.copy()

    # Inverse transforming imputed variables
    ord_inv_transforming(data)

    # Assigning transformed variable back to the dataframe
    df[miss] = data[miss]

    # One hot transforming and scaling the variables
    df = pd.DataFrame(trans_onehot_scale.transform(df), columns=List)

    # Filtering top 10 features selected by Sklearn RFE function
    df = df[rfe_vars]

    # Predicting the data point
    y = model.predict(df)

    return y