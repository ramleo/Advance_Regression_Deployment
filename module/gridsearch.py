def gs_lasso(data=None, y=None):
    '''Calculating optimal alpha using grid search and repeated kfolds'''

    # Importing relevant packages
    from sklearn.model_selection import GridSearchCV, RepeatedKFold
    from sklearn.linear_model import Lasso

    # Initiating the ridge
    lasso_gs = Lasso(max_iter=5000, random_state=7)

    # Hyper-parameter tuning with various lambda/alpha values
    param = {'alpha': [0, 0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]}

    # Initiating Repeated KFolds
    kfolds = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    # Initiating the GridSearch
    gs_lasso = GridSearchCV(estimator=lasso_gs,
                            param_grid=param,
                            cv=kfolds,
                            n_jobs=-1,
                            scoring=['neg_mean_absolute_error', 'neg_median_absolute_error', 'r2'],
                            return_train_score=True, refit=False)

    # Fitting the GridSearch to the train data set
    result = gs_lasso.fit(data, y)
    return result