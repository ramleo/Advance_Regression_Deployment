def import_data_split(file=None):
    '''Importing dataset and splitting into train-test data and returning copy of train-test datasets'''
    import pandas
    from sklearn.model_selection import train_test_split

    housing = pandas.read_csv(r'C:\Users\DA1041TU\Documents\UpGrad\Data\HousingDataset_Australia' + file)

    X = housing.drop(['SalePrice','Id'], axis=1)
    y = housing['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    return X_train, X_test, y_train, y_test

