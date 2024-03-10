#encoding: utf-8
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
    
@pytest.fixture(scope='session')
def testclientfixture():
    with TestClient(app) as client:
        yield client
    
def test_base_predict(testclientfixture):
    for model_name in ModelEnum:
        if model_name == ModelEnum.hybrid:
            continue
        resp = testclientfixture.post(
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


@pytest.mark.parametrize("input, expected_output", [
    ("עשרות אנשים מגיעים מתאילנד לישראל כשהם נרשמים כמתנדבים, אך למעשה משמשים עובדים שכירים זולים .",
     ['O', 'O', 'O', 'S-GPE', 'S-GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
    
    ("תופעה זו התבררה אתמול בוועדת העבודה והרווחה של הכנסת, שדנה בנושא העסקת עובדים זרים .",
     ['O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'E-ORG', 'O', 'S-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
    
    ("כמו כן, תציב הצעת החוק עונשי מאסר והטלת קנסות כבדים למי שיעסיק עובדים זרים בלא רשיון .",
     ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
    
    ("אני גם מוכן להסתכן ולשער כי ההורה, או קבוצת ההורים שהתנגדה לטיול, מסתתרים תחת המעטה של טיעון פוליטי נבוב ולמעשה הם פוחדים לשלוח את ילדיהם שמא איזה ערבי ינעץ בהם סכין בגב .",
     ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
    
    ("איש בטקסס לא פיקפק שיריבה הרפובליקאי, קלייטון ויליאמס, חוואי ואיש נפט, יביס אותה בקלות .",
     ['O', 'S-GPE', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'E-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
])
def test_standard_predict(testclientfixture, input, expected_output):
    # with TestClient(app) as client:
    resp = testclientfixture.post(
        "/predict",
        json={"text": input, "model": ModelEnum.token_single}
    )
    assert resp.status_code == 200
    labels = [x["label"] for x in resp.json()["prediction"][0]]
    assert labels == expected_output

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