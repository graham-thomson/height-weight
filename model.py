import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train(seed=None):
    df = pd.read_csv(
        "./data/weight-height.csv",
        header=0,
        doublequote=True
    )
    df["Gender"] = df["Gender"].apply(lambda x: 1. if x == "Male" else 0.)

    x_train, x_test, y_train, y_test = train_test_split(
        df[["Height", "Weight"]],
        df["Gender"],
        test_size=0.33,
        random_state=seed
    )

    lr = LogisticRegression(solver="lbfgs")

    pipeline = PMMLPipeline([
        ("classifier", lr)
    ])

    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    acc, prec, rec = accuracy_score(y_test, preds), \
                     precision_score(y_test, preds), \
                     recall_score(y_test, preds)

    model_perf = {
        "Accuracy": f"{acc:.02f}",
        "Precision": f"{prec:.02f}",
        "Recall": f"{rec:.02f}"
    }

    sklearn2pmml(
        pipeline=pipeline,
        pmml="height_weight_lr_model.pmml"
    )

    return {"model": lr, "model_perf": model_perf}


if __name__ == '__main__':
    print(train(seed=88))