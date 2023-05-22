import sklearn.metrics

def get_f1_score(gold, inference, average="macro"):
  return sklearn.metrics.f1_score(gold, inference, average=average)

def train_and_evaluate(model, training_set, evaluation_set, key, plot_confusion_matrix=True):
  model.train(training_set)
  inference = model.predict(evaluation_set)
  if plot_confusion_matrix:
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(evaluation_set[key], inference, xticks_rotation="vertical")
  print(f"f1 score for {key}: {get_f1_score(evaluation_set[key], inference)}")