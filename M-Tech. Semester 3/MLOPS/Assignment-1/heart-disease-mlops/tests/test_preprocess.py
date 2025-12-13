from src.preprocess import load_data

def test_load_data_not_empty():
    df = load_data("data/heart.csv")
    assert df.shape[0] > 0
    assert df.shape[1] == 14

from src.preprocess import drop_missing

def test_drop_missing_reduces_rows():
    df = load_data("data/heart.csv")
    df_clean = drop_missing(df)
    assert df_clean.shape[0] < df.shape[0]

from src.preprocess import fix_target

def test_target_is_binary():
    df = load_data("data/heart.csv")
    df = drop_missing(df)
    df = fix_target(df)

    assert set(df["target"].unique()).issubset({0, 1})


from src.preprocess import (
    load_data,
    drop_missing,
    fix_target,
    split_features_target,
    build_preprocessor
)

def test_preprocessing_pipeline_output_shape():
    # Load and clean data
    df = load_data("data/heart.csv")
    df = drop_missing(df)
    df = fix_target(df)

    # Split features and target
    X, y = split_features_target(df)

    # Build and apply preprocessing pipeline
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Assertions
    assert X_transformed.shape[0] == X.shape[0]   # same number of rows
    assert X_transformed.shape[1] == X.shape[1]   # same number of features

