def evaluate_model(classifier=None, pred_data=None, actual_y=None):
    '''Prints metrics such as R Square, Mean Absolute Error,
    Median Absolute Error and Root Mean Squared Error'''

    # Importing relevant packages
    from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
    import numpy as np
    # Predicting target variable
    lasso_ypred = classifier.predict(pred_data)
    print(f'R Square is :{round(r2_score(actual_y, lasso_ypred), 3)}')
    print(f'Mean Absolute Error is: {round(mean_absolute_error(actual_y, lasso_ypred), 3)}')
    print(f'Median Absolute Error is: {round(median_absolute_error(actual_y, lasso_ypred), 3)}')
    print(f'Root Mean Squared Error is: {round(np.sqrt(mean_squared_error(actual_y, lasso_ypred)), 3)}')