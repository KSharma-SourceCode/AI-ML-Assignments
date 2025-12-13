from src.train import train
from sklearn.pipeline import Pipeline

def test_training_produces_model_with_reasonable_accuracy():
    pipeline, acc = train()

    # Model object checks
    assert pipeline is not None
    assert isinstance(pipeline, Pipeline)

    # Accuracy sanity check (not exact match!)
    assert acc >= 0.75
