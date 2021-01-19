from sklearn.model_selection import KFold
from CNN.helpers import create_sequential_model, prepare_training_and_testing_data, plt_history, \
    apply_cross_validation_and_evaluate, accuracy_summary


def run_cnn(arc_number):
    # All images have the same square size of 28×28 pixels.
    # the images are grayscale.
    (train_x, train_y), (test_x, test_y) = prepare_training_and_testing_data()

    nkfold = 3
    results, histories, bestModel = apply_cross_validation_and_evaluate(train_x, train_y, nkfold ,arc_number)

    plt_history(histories)

    _,acc = bestModel.evaluate(test_x,test_y,verbose=0)
    print("\n\nModel Accuracy of the best model found in the cross validation is:",acc)
    accuracy_summary(results)

run_cnn(1)