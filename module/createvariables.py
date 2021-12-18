def create_variables(data=None):
    '''Creating variables for preprocessing of the data'''
    # Importing relevant packages
    import numpy as np
    from module.datacleaner import data_cleaning

    # Taking a copy
    data = data.copy()
    # Creating list of ordinal variables
    ord_vars = ['ExterQual', 'HeatingQC', 'ExterCond', 'KitchenQual', 'PoolQC', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'FireplaceQu',
                'GarageQual', 'GarageCond', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 'LotShape', 'SaleCondition',
                'Fence']

    # Making list of categories in a variable to use it in Ordinal encoding

    eq = ['Fa', 'TA', 'Gd', 'Ex']
    hq = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    ec = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    kq = ['Fa', 'TA', 'Gd', 'Ex']
    pc = ['No Pool', 'Fa', 'Gd', 'Ex']
    bq = ['No Basement', 'Fa', 'TA', 'Gd', 'Ex']
    bc = ['No Basement', 'Po', 'Fa', 'TA', 'Gd']
    be = ['No Basement', 'No', 'Mn', 'Av', 'Gd']
    fq = ['No Fireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    gq = ['No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    gc = ['No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    bft1 = ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
    bft2 = ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
    gf = ['No Garage', 'Unf', 'RFn', 'Fin']
    ls = ['IR3', 'IR2', 'IR1', 'Reg']
    sc = ['Partial', 'Family', 'Alloca', 'AdjLand', 'Abnorml', 'Normal']
    f = ['No Fence', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']

    list_ord_vars = [eq, hq, ec, kq, pc, bq, bc, be, fq, gq, gc, bft1, bft2, gf, ls, sc, f]

    # Creating list of onehot and numeric variables
    onehot_vars = []
    num_vars = []
    df = data_cleaning(data=data)
    for i in df.columns:
        if (df[i].dtype == 'O') and (i not in ord_vars):
            onehot_vars.append(i)
        elif (df[i].dtype != 'O') and (i not in ord_vars):
            num_vars.append(i)
    scale_vars = ord_vars + num_vars

    # Creating list of total variables in the dataset
    total_vars = ['LotFrontage', 'MasVnrArea', 'MasVnrType', 'Electrical']

    for i in df.columns:
        if i not in total_vars:
            total_vars.append(i)

    # List of onehot encoded variables
    after_onehot = [np.array(['C (all)', 'FV', 'RH', 'RL', 'RM'], dtype=object),
                    np.array(['Grvl', 'Pave'], dtype=object),
                    np.array(['Grvl', 'No alley access', 'Pave'], dtype=object),
                    np.array(['Bnk', 'HLS', 'Low', 'Lvl'], dtype=object),
                    np.array(['AllPub', 'NoSeWa'], dtype=object),
                    np.array(['Corner', 'CulDSac', 'FR2', 'FR3', 'Inside'], dtype=object),
                    np.array(['Gtl', 'Mod', 'Sev'], dtype=object),
                    np.array(['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
                              'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                              'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
                              'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber',
                              'Veenker'], dtype=object),
                    np.array(['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNe',
                              'RRNn'], dtype=object),
                    np.array(['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNn'],
                             dtype=object),
                    np.array(['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'], dtype=object),
                    np.array(['1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story',
                              'SFoyer', 'SLvl'], dtype=object),
                    np.array(['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'], dtype=object),
                    np.array(['ClyTile', 'CompShg', 'Metal', 'Tar&Grv', 'WdShake', 'WdShngl'],
                             dtype=object),
                    np.array(['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CemntBd', 'HdBoard',
                              'ImStucc', 'MetalSd', 'Plywood', 'Stone', 'Stucco', 'VinylSd',
                              'Wd Sdng', 'WdShing'], dtype=object),
                    np.array(['AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace', 'CmentBd', 'HdBoard',
                              'ImStucc', 'MetalSd', 'Plywood', 'Stone', 'Stucco', 'VinylSd',
                              'Wd Sdng', 'Wd Shng'], dtype=object),
                    np.array(['BrkCmn', 'BrkFace', 'None', 'Stone'], dtype=object),
                    np.array(['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'], dtype=object),
                    np.array(['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'], dtype=object),
                    np.array(['N', 'Y'], dtype=object),
                    np.array(['FuseA', 'FuseF', 'FuseP', 'Mix', 'SBrkr'], dtype=object),
                    np.array(['Maj1', 'Maj2', 'Min1', 'Min2', 'Mod', 'Typ'], dtype=object),
                    np.array(['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd',
                              'No Garage'], dtype=object),
                    np.array(['N', 'P', 'Y'], dtype=object),
                    np.array(['COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth', 'WD'],
                             dtype=object)]

    # Renaming onehot encoded variables using names of original variables
    list2 = [i + '_' + j for i, j in zip(onehot_vars, after_onehot)]

    list1 = []
    for i in list2:
        list1.extend(i)

    return ord_vars, list_ord_vars, onehot_vars, num_vars, total_vars, list1