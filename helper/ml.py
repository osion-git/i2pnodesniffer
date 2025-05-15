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
    # Dynamically detect number of classes and smallest class
    class_counts = df[target_col].value_counts()
    n_min = class_counts.min()

    # Extract random entries for each class
    balanced_parts = [
        df[df[target_col] == cls].sample(n=n_min, random_state=seed)
        for cls in class_counts.index
    ]

    df_balanced = (
        pd.concat(balanced_parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return df_balanced
