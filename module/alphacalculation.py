def cal_alpha(data=None, y=None):
    '''Finding optimal alpha value as a result og grid search'''

    # Import relevant packages
    import pandas as pd
    from module.gridsearch import gs_lasso

    # Calling user defined grid search function
    lasso_model = gs_lasso(data=data, y=y)

    # Creating dataframe of gris search result
    df = pd.DataFrame(lasso_model.cv_results_)

    # Saving index of best value of the metric
    max_idx = df['mean_test_neg_median_absolute_error'].argmax()

    # Saving parameter
    params = df[max_idx:max_idx + 1]['params']

    # Saving value of parameter
    alpha = params.values[0]['alpha']

    return alpha