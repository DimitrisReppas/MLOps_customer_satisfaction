import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader

def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    high_level_image = Image.open("_assets/high_level_overview.png")
    st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. We will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum error requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )

    # Use st.form to group inputs together
    with st.form(key='input_form'):
        st.markdown("### Input Features (product details)")
        
        payment_sequential = st.slider("Payment Sequential", min_value=1, max_value=10, value=1) # minimum and maximum values for the sliders were hypothetical. You can change them if you want!
        payment_installments = st.slider("Payment Installments", min_value=1, max_value=24, value=1)
        payment_value = st.slider("Payment Value", min_value=0.0, max_value=1000.0, value=100.0)
        price = st.slider("Price", min_value=0.0, max_value=1000.0, value=100.0)
        freight_value = st.slider("Freight Value", min_value=0.0, max_value=100.0, value=10.0)
        product_name_length = st.slider("Product Name Length", min_value=0, max_value=100, value=50)
        product_description_length = st.slider("Product Description Length", min_value=0, max_value=1000, value=100)
        product_photos_qty = st.slider("Product Photos Quantity", min_value=0, max_value=20, value=5)
        product_weight_g = st.slider("Product Weight (g)", min_value=0, max_value=20000, value=500)
        product_length_cm = st.slider("Product Length (cm)", min_value=0, max_value=100, value=10)
        product_height_cm = st.slider("Product Height (cm)", min_value=0, max_value=100, value=10)
        product_width_cm = st.slider("Product Width (cm)", min_value=0, max_value=100, value=10)

        # Add a submit button for the form
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            # Run the pipeline here if needed

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        service.start(timeout=10)  # should be a NOP if already started
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfactory rate (range between 0 - 5) with given product details is: {}".format(
                pred
            )
        )

    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree-based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        image = Image.open("_assets/feature_importance_gain.png")
        st.image(image, caption="Feature Importance Gain")

if __name__ == "__main__":
    main()