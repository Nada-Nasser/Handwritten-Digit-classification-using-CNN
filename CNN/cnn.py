from sklearn.model_selection import KFold
from CNN.helpers import create_sequential_model, prepare_training_and_testing_data, plt_history, \
    apply_cross_validation_and_evaluate, accuracy_summary


def run_cnn(arc_number):
    # All images have the same square size of 28×28 pixels.
    # the images are grayscale.
    (train_x, train_y), (test_x, test_y) = prepare_training_and_testing_data()

    model = create_sequential_model(arc_number) # compile and build CNN model
    kfold = KFold(n_splits=3, shuffle=True, random_state=1)

    results, histories, model = apply_cross_validation_and_evaluate(model, train_x, train_y, kfold)

    plt_history(histories)
    accuracy_summary(results)

run_cnn(1)