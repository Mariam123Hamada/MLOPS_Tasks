# src/prodml/train.py

import pickle


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import create_features, prepare_feature_dicts


def train_model() -> dict[str, float]:
    """Train the trip duration prediction model."""

    df = load_data()

    df = create_features(df)

    df_train, df_val = split_data(df)

    train_dicts = prepare_feature_dicts(df_train)
    val_dicts = prepare_feature_dicts(df_val)

    dv = DictVectorizer()

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred)

    mae = mean_absolute_error(
        y_val,
        y_pred,
    )

    settings.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(settings.model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "vectorizer": dv,
            },
            f,
        )

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    return {
        "rmse": float(rmse),
        "mae": float(mae),
    }


def main() -> None:
    """CLI entry point."""

    train_model()


if __name__ == "__main__":
    main()
