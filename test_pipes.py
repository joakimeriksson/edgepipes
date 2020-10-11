import edgepipes

def test_load():
    # test creation and load a file
    pipeline = edgepipes.Pipeline()
    with open("graphs/edge_detection.pbtxt", "r") as f:
        txt = f.read()
    assert(len(txt) > 0)
