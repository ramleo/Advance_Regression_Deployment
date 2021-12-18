def prep_test(features=None, alpha=None, data=None, y=None, pred_data=None, pipeline=None):
    '''Preparing the model for final prediction'''

    # Importing relevant packages
    from sklearn.linear_model import Lasso
    # Initiating model
    lasso_model = Lasso(alpha=alpha, max_iter=100000, random_state=7)
    lasso_model.fit(data, y)
    # Transforming dataset
    pred_data = pipeline.transform(pred_data)
    pred_data = pred_data[features]

    return lasso_model, pred_data