"""Train the PRPD defect classifier and save pd_classifier.pkl.

The training set is synthetic: PRPD phase profiles generated from the textbook signature of each
defect type (pdguard_core.generate_pattern). It must be re-trained on labelled field data before
any accuracy figure is quoted to a client.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import pdguard_core as pc

X, y = pc.make_training_set(n_per_class=1500, seed=11)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
model = RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=2, n_jobs=-1, random_state=0)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print(classification_report(yte, pred, digits=3))
print('labels:', list(model.classes_))
print(confusion_matrix(yte, pred, labels=model.classes_))

# final model on all data
model.fit(X, y)
joblib.dump(model, 'pd_classifier.pkl', compress=3)
print('saved pd_classifier.pkl')
