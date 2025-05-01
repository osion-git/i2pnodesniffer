import pandas as pd


def undersample_df(df, target_col='prediction', seed=42):
    """
    Performs undersampling to balance the dominant class
    and returns a balanced DataFrame.

    :param df: pandas.DataFrame with a binary target column
    :param target_col: name of the label column
    :param seed: random seed for reproducibility
    :return: balanced DataFrame
    """
    # Split into both classes
    df0 = df[df[target_col] == 0]
    df1 = df[df[target_col] == 1]

    # Size of the minority class
    n = min(len(df0), len(df1))

    # Randomly sample from each class (dominant class is reduced)
    df0_samp = df0.sample(n, random_state=seed)
    df1_samp = df1.sample(n, random_state=seed)

    # Combine
    df_bal = pd.concat([df0_samp, df1_samp]) \
        .sample(frac=1, random_state=seed) \
        .reset_index(drop=True)

    return df_bal
