from helpers import create_sequential_model,prepare_training_and_testing_data, plt_history


def run_cnn(arc_number):
    # All images have the same square size of 28×28 pixels.
    # the images are grayscale.
    (train_x, train_y), (test_x, test_y) = prepare_training_and_testing_data()

    model = create_sequential_model(arc_number) # compile and build CNN model

    history = model.fit(train_x, train_y, epochs=10,
                        validation_data=(test_x, test_y)) # train the model

    plt_history(history)

run_cnn(1)