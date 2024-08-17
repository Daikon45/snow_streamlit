try:
    from pycaret.regression import load_model, predict_model
    print("Import successful")
except ImportError as e:
    print("Import error:", e)
    