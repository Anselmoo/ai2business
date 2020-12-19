from unittest import mock

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from ai2business.ai_engines import automl_neural_network as an


@mock.patch("ai2business.ai_engines.automl_neural_network.AutoMLFit")
def test_interest_over_time(AutoMLFit):
    X, y = load_boston(return_X_y=True)
    x_train, y_train, x_test, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    context = an.AutoMLPipeline(an.TimeseriesForecaster())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()

    assert an.AutoMLFit.is_called


def test_run():
    X, y = load_boston(return_X_y=True)
    x_train, y_train, x_test, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
    )
    context = an.AutoMLPipeline(an.TimeseriesForecaster())
    context.run_automl()
    context.train = an.AutoMLFit(x_train, y_train, batch_size=32, epochs=1)
    context.run_automl()
    context.train = an.AutoMLEvaluate(x_test, y_test, batch_size=32)
    context.run_automl()
    # assert an.AutoMLFit.is_called
