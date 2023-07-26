import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras

st.title('Problem Rating Predictor')

# Load models
identification_model = keras.models.load_model('models/identification_model.h5')
resolution_model = keras.models.load_model('models/resolution_model.h5')

# Define the inputs for the models
inputs = ['content_word_count', 'content_character_count', 'AccountDeactivated', 'AccountLimit', 'AccountNotActive',
          'CodeNotDetected', 'Coupon', 'CouponMultipleUsers', 'GamekeyActivated', 'LogicError', 'PVT',
          'RedemptionFail', 'RedemptionSuccess', 'SupportSeeNewerTicket', 'Token', 'WebGLonCoupon',
          'couponAlreadyAssigned', 'couponAlreadyUsed', 'couponExpired', 'couponLimitReached', 'couponNotStarted',
          'couponNotValid', 'couponWaitingActivation', 'hardwareNotDetected', 'proofOfPurchase', 'pvtDownloadRequired',
          'systemGuardExtraPermanentBlock', 'systemNotValidTemporaryBlock', 'Account-Activation', 'Account-Limit',
          'Account-Login', 'Account-Password', 'Alert-ProofofPurchaseRequired', 'Coupon-AlreadyUsed',
          'Coupon-Expiration', 'Coupon-NotValid', 'Coupon-ProofofPurchase', 'Coupon-Received', 'GameProvider-Steam',
          'Other-IHaveNotInstalled', 'PVT-DNS', 'PVT-General', 'PVT-HowTo', 'PVT-Linux',
          'ProgramPartner-AMDPleaseProvideCoupon', 'ProgramPartner-Amazon', 'ProgramPartner-DidNotReceiveCoupon',
          'PromotionRules-HardwareNotDetected', 'WrongPortal-DriverSupport', 'WrongPortal-Error195']

# Create a dictionary to hold the inputs
data = {}

# Create the Streamlit inputs
for input in inputs:
    data[input] = st.sidebar.slider(input, 0, 1000, 50)

# Convert the dictionary to a DataFrame
df = pd.DataFrame([data])

# Perform the prediction
preds_identification = np.argmax(identification_model.predict(df), axis=1)
preds_resolution = np.argmax(resolution_model.predict(df), axis=1)

# Display the predictions
st.write(f'Prediction for problem_rating_identification: {preds_identification[0]}')
st.write(f'Prediction for problem_rating_resolution: {preds_resolution[0]}')
