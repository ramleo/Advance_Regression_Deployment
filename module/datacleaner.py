
def data_cleaning(data=None):
    '''This function cleans the dataset'''

    # Creating 'Age' column from 'YearBuilt'
    data['Age(in_Years)'] = 2021-data['YearBuilt']

    # Creating 'BuiltRemoddDiff' by taking difference between 'YearRemodAdd' and 'YearBuilt'
    data['BuiltRemoddDiff'] = data['YearRemodAdd']-data['YearBuilt']

    # Creating 'SaleRemoddDiff' by taking difference between 2021 and 'YearRemodAdd'
    data['SaleRemoddDiff'] = 2021-data['YearRemodAdd']

    # Converting 'NA' values to its relevant Category as they are not actually 'Null Values'
    data.loc[data['Alley'].isnull(), 'Alley'] = 'No alley access'
    data.loc[data['FireplaceQu'].isnull(), 'FireplaceQu'] = 'No Fireplace'
    data.loc[data['PoolQC'].isnull(), 'PoolQC'] = 'No Pool'
    data.loc[data['Fence'].isnull(), 'Fence'] = 'No Fence'
    data.loc[data['BsmtQual'].isnull(), 'BsmtQual'] = 'No Basement'
    data.loc[data['BsmtCond'].isnull(), 'BsmtCond'] = 'No Basement'
    data.loc[data['BsmtExposure'].isnull(), 'BsmtExposure'] = 'No Basement'
    data.loc[data['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'No Basement'
    data.loc[data['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 'No Basement'
    data.loc[data['GarageType'].isnull(), 'GarageType'] = 'No Garage'
    data.loc[data['GarageFinish'].isnull(), 'GarageFinish'] = 'No Garage'
    data.loc[data['GarageQual'].isnull(), 'GarageQual'] = 'No Garage'
    data.loc[data['GarageCond'].isnull(), 'GarageCond'] = 'No Garage'

    # Dropping columns with more than 90% NA values and unnecessary columns
    data = data.drop(['YearBuilt','MiscFeature', 'YrSold', 'GarageYrBlt', 'YearRemodAdd', 'MoSold'], axis=1)
    return data