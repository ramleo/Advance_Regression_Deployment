def pred_train(data=None, y=None):
    '''Predicting train dataset'''

    # Importing relevant packages
    from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
    import numpy as np
    from sklearn.linear_model import Lasso

    # Features selected using 'f_regression' feature selection method
    freg_feat = ['ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish', 'OverallCond',
                 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'GarageCars',
                 'GarageArea']

    # Created train dataset using 'f_regression' feature selection method
    train_freg = data[freg_feat]

    # Instantiating 'Ridge' and 'Lasso' models
    model_lasso = Lasso(max_iter=5000, random_state=7)

    # Prediction for Lasso Regression
    ypred_train = model_lasso.fit(train_freg, y).predict(train_freg)

    print(f'R Square is :{round(r2_score(actual_y, lasso_ypred), 3)}')
    print(f'Mean Absolute Error is: {round(mean_absolute_error(actual_y, lasso_ypred), 3)}')
    print(f'Median Absolute Error is: {round(median_absolute_error(actual_y, lasso_ypred), 3)}')
    print(f'Root Mean Squared Error is: {round(np.sqrt(mean_squared_error(actual_y, lasso_ypred)), 3)}')