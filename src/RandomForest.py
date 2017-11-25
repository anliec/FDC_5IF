import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.utils import read_new_csv, csv_set_to_sklearn_batch, expended_read_new_csv

if __name__ == "__main__":
    csv_dict = expended_read_new_csv("data/train2.csv")
    # csv_dict = read_new_csv("data/train2.csv")

    input_batch, output_batch = csv_set_to_sklearn_batch(csv_dict)

    clf = RandomForestClassifier(random_state=None, class_weight="balanced")
    pipe = Pipeline(steps=[('tree', clf)])

    # Prediction
    max_depth = [10, 15, 20, 25]
    min_samples_split = [2, 3, 4, 5]
    n_estimators = [90, 100, 110, 120]

    clf.fit(X=np.array(input_batch),
            y=np.array(output_batch))

    estimator = GridSearchCV(pipe,
                             dict(tree__max_depth=max_depth,
                                  tree__min_samples_split=min_samples_split,
                                  tree__n_estimators=n_estimators),
                             n_jobs=-1, cv=4, verbose=1)
    estimator.fit(X=input_batch,
                  y=output_batch)

    print(estimator.best_params_)
    print(estimator.best_score_)

    csv_test_dict = read_new_csv("data/test2f.csv")
    input_test_batch, _ = csv_set_to_sklearn_batch(csv_test_dict)

    prediction = estimator.predict(X=input_test_batch)

    with open("submission.csv", "w") as submission:
        submission.write("RowId,prediction")
        idx = 1
        for item in prediction:
            submission.write("\n%i,%s" % (idx, item))
            idx += 1


