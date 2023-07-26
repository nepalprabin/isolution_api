from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow import keras

import json
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel


# Create a FastAPI instance
app = FastAPI()

# Load models
identification_model = keras.models.load_model('models/identification_model.h5')
resolution_model = keras.models.load_model('models/resolution_model.h5')

# Create a Pydantic model to validate the data in the request payload
class Item(BaseModel):
    content_word_count: int
    content_character_count: int
    AccountDeactivated: int
    AccountLimit: int
    AccountNotActive: int
    CodeNotDetected: int
    Coupon: int
    CouponMultipleUsers: int
    GamekeyActivated: int
    LogicError: int
    PVT: int
    RedemptionFail: int
    RedemptionSuccess: int
    SupportSeeNewerTicket: int
    Token: int
    WebGLonCoupon: int
    couponAlreadyAssigned: int
    couponAlreadyUsed: int
    couponExpired: int
    couponLimitReached: int
    couponNotStarted: int
    couponNotValid: int
    couponWaitingActivation: int
    hardwareNotDetected: int
    proofOfPurchase: int
    pvtDownloadRequired: int
    systemGuardExtraPermanentBlock: int
    systemNotValidTemporaryBlock: int
    Account_Activation: int
    Account_Limit: int
    Account_Login: int
    Account_Password: int
    Alert_ProofofPurchaseRequired: int
    Coupon_AlreadyUsed: int
    Coupon_Expiration: int
    Coupon_NotValid: int
    Coupon_ProofofPurchase: int
    Coupon_Received: int
    GameProvider_Steam: int
    Other_IHaveNotInstalled: int
    PVT_DNS: int
    PVT_General: int
    PVT_HowTo: int
    PVT_Linux: int
    ProgramPartner_AMDPleaseProvideCoupon: int
    ProgramPartner_Amazon: int
    ProgramPartner_DidNotReceiveCoupon: int
    PromotionRules_HardwareNotDetected: int
    WrongPortal_DriverSupport: int
    WrongPortal_Error195: int

@app.post("/predict")
def predict(item: Item):
    # Convert the request payload to a pandas DataFrame
    input_data = pd.DataFrame([dict(item)])

    # Ensure the input data is in the correct format and type
    input_data = input_data.fillna(0).astype('int')

    # Inference for 'problem_rating_identifitication'
    preds_identification = np.argmax(identification_model.predict(input_data), axis = 1)
    print(preds_identification)

    # Inference for 'problem_rating_resolution'
    preds_resolution = np.argmax(resolution_model.predict(input_data), axis = 1)
    print(preds_resolution)
    return {
        "problem_rating_identifitication": preds_identification.tolist(),
        "problem_rating_resolution": preds_resolution.tolist()
    }


@app.post("/json_predict")
async def upload_json(file: UploadFile = File(...)):
    # Load the contents of the file
    contents = await file.read()

    # Load the contents as JSON
    input_data = json.loads(contents)

    # Convert to DataFrame
    input_data = pd.DataFrame([input_data])

    # Ensure the input data is in the correct format and type
    input_data = input_data.fillna(0).astype('int')

    # Perform the prediction
    preds_identification = np.argmax(identification_model.predict(input_data), axis = 1)
    preds_resolution = np.argmax(resolution_model.predict(input_data), axis = 1)

    return {
        "problem_rating_identification": preds_identification.tolist(),
        "problem_rating_resolution": preds_resolution.tolist()
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)