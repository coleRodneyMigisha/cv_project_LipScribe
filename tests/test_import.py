def test_package_imports():
    import lipscribe
    from lipscribe import data, preprocessing, models, decoding, evaluation, app
    assert lipscribe.__version__