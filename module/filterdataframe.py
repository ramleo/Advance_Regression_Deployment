def filter_df_sel_feat(data=None, pipeline=None):
    '''Filtering the dataset using features selected by feature selection model'''

    # Features selected using 'f_regression' feature selection method
    freg_feat = ['ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish', 'OverallCond',
                   'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'GarageCars',
                   'GarageArea']
    # Performing pipeline transform on validation dataset
    data = pipeline.transform(data)
    # Creating X_val dataset using selected features
    data = data[freg_feat]
    return freg_feat, data