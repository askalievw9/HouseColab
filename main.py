import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib



app = FastAPI()

model = joblib.load('gb_model (2).pkl')
scaler = joblib.load('scaler (3).pkl')



class PersonSchema(BaseModel):
    OverallQual: int
    Neighborhood: str
    GrLivArea: int
    GarageCars: int
    GarageArea: int
    TotalBsmtSF: int
    firstFlrSF: int
    FullBath: int
    YearBuilt: int
    YearRemodAdd: int
    TotRmsAbvGrd: int


@app.post('/predict')
async def predict(person: PersonSchema):
    person_dict = person.dict()

    neighborhood = person_dict.pop('Neighborhood')
    neighborhood_0_1 = [

        1 if neighborhood == 'Blueste' else 0,
        1 if neighborhood == 'BrDale' else 0,
        1 if neighborhood == 'BrkSide' else 0,
        1 if neighborhood == 'ClearCr' else 0,
        1 if neighborhood == 'CollgCr' else 0,
        1 if neighborhood == 'Crawfor' else 0,
        1 if neighborhood == 'Edwards' else 0,
        1 if neighborhood == 'Gilbert' else 0,
        1 if neighborhood == 'IDOTRR' else 0,
        1 if neighborhood == 'MeadowV' else 0,
        1 if neighborhood == 'Mitchel' else 0,
        1 if neighborhood == 'NAmes' else 0,
        1 if neighborhood == 'NPkVill' else 0,
        1 if neighborhood == 'NWAmes' else 0,
        1 if neighborhood == 'NoRidge' else 0,
        1 if neighborhood == 'NridgHt' else 0,
        1 if neighborhood == 'OldTown' else 0,
        1 if neighborhood == 'SWISU' else 0,
        1 if neighborhood == 'Sawyer' else 0,
        1 if neighborhood == 'SawyerW' else 0,
        1 if neighborhood == 'Somerst' else 0,
        1 if neighborhood == 'StoneBr' else 0,
        1 if neighborhood == 'Timber' else 0,
        1 if neighborhood == 'Veenker' else 0,
    ]

    features = list(person_dict.values()) + neighborhood_0_1
    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]
    print(model.predict(scaled))

    return {"approved": round(pred)}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)