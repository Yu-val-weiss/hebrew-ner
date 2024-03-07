import os
from fastapi.testclient import TestClient
from ner_app import app, NamedTemporary, ModelEnum
import pytest

testclient = TestClient(app)

def test_home():
    response = testclient.get("/")
    assert response.status_code == 200
    
def test_healthcheck():
    response = testclient.get("/healthcheck")
    assert response.status_code == 200
    
def test_base_predict():
    with TestClient(app) as client:
        for model_name in ModelEnum:
            resp = client.post(
                "/predict",
                json={"text": "משפת דוגמה . ", "model": model_name}
            )
            assert resp.status_code == 200
            assert resp.json() == {
                "prediction": [
                    [
                        {
                            "token": "משפת",
                            "label": "O"
                        }, 
                        {
                            "token": "דוגמה",
                            "label": "O"
                        },
                        {
                            "token": ".",
                            "label": "O"
                        },
                    ]
                ]
            }

def test_temp_file():
    with NamedTemporary() as tmpf:
        assert "heb-ner-tmp-" in tmpf
        assert os.path.exists(tmpf)
    assert not os.path.exists(tmpf)

def test_tokenize():
    text = {"text": "גנן גידל דגן בגן."}
    response = testclient.post("/tokenize", json=text)
    assert response.status_code == 200
    data = response.json()
    # verify shape
    assert isinstance(data, list)
    assert isinstance(data[0], list)
    assert isinstance(data[0][0], str)
    # verify content
    assert data[0] == ['גנן', 'גידל', 'דגן', 'בגן', '.']