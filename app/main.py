from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from .model_loaders import load_keras_model, load_pipeline
from .models import InputData
import pandas as pd
from scipy.sparse import hstack
import warnings

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils._tags")

# Inicializar la aplicación
app = FastAPI()

# Ruta de los modelos
MODEL_PATH = "./models/best_model2.h5"
PIPELINE_PREPROCESSOR_PATH = "./models/pipeline_preprocessor.joblib"
TFIDF_DESC_PATH = "./models/tfidf_desc.joblib"
TFIDF_EMP_PATH = "./models/tfidf_emp.joblib"

CVEC_DESC_PATH = "./models/cvec_desc.joblib"
CVEC_EMP_PATH = "./models/cvec_emp.joblib"
FEATURE_SELECTOR_PATH = "./models/feature_selector.joblib"
XGB_MODEL_PATH = "./models/xgb_model.joblib"
COLUMNS_TRANSFORMED_PATH = "./models/columns_transformed.joblib"
SELECTED_COLUMNS_PATH = "./models/selected_columns.joblib"

# Cargar el modelo y los pipelines
model = load_keras_model(MODEL_PATH)
pipeline_preprocessor = load_pipeline(PIPELINE_PREPROCESSOR_PATH)
tfidf_desc = load_pipeline(TFIDF_DESC_PATH)
tfidf_emp = load_pipeline(TFIDF_EMP_PATH)
cvec_desc = load_pipeline(CVEC_DESC_PATH)
cvec_emp = load_pipeline(CVEC_EMP_PATH)
feature_selector = load_pipeline(FEATURE_SELECTOR_PATH)
xgb_model = load_pipeline(XGB_MODEL_PATH)
columns_transformed = load_pipeline(COLUMNS_TRANSFORMED_PATH)
selected_columns = load_pipeline(SELECTED_COLUMNS_PATH)
class_labels = ['fully paid', 'charged', 'current', 'default', 'late days',
                'grace period', 'meet credit policy status fully paid',
                'meet credit policy status charged', 'issued']
# Endpoint para realizar predicciones
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Crear un DataFrame con los datos de entrada
        input_data = {
            "loan_amnt": [data.loan_amnt],
            "funded_amnt": [data.funded_amnt],
            "funded_amnt_inv": [data.funded_amnt_inv],
            "term": [data.term],
            "int_rate": [data.int_rate],
            "installment": [data.installment],
            "grade": [data.grade],
            "sub_grade": [data.sub_grade],
            "emp_length": [data.emp_length],
            "home_ownership": [data.home_ownership],
            "annual_inc": [data.annual_inc],
            "verification_status": [data.verification_status],
            "issue_d": [data.issue_d],
            "loan_status": [data.loan_status],
            "pymnt_plan": [data.pymnt_plan],
            "purpose": [data.purpose],
            "title": [data.title],
            "zip_code": [data.zip_code],
            "addr_state": [data.addr_state],
            "dti": [data.dti],
            "delinq_2yrs": [data.delinq_2yrs],
            "earliest_cr_line": [data.earliest_cr_line],
            "inq_last_6mths": [data.inq_last_6mths],
            "mths_since_last_delinq": [data.mths_since_last_delinq],
            "open_acc": [data.open_acc],
            "pub_rec": [data.pub_rec],
            "revol_bal": [data.revol_bal],
            "revol_util": [data.revol_util],
            "total_acc": [data.total_acc],
            "initial_list_status": [data.initial_list_status],
            "out_prncp": [data.out_prncp],
            "out_prncp_inv": [data.out_prncp_inv],
            "total_pymnt": [data.total_pymnt],
            "total_pymnt_inv": [data.total_pymnt_inv],
            "total_rec_prncp": [data.total_rec_prncp],
            "total_rec_int": [data.total_rec_int],
            "total_rec_late_fee": [data.total_rec_late_fee],
            "recoveries": [data.recoveries],
            "collection_recovery_fee": [data.collection_recovery_fee],
            "last_pymnt_d": [data.last_pymnt_d],
            "last_pymnt_amnt": [data.last_pymnt_amnt],
            "next_pymnt_d": [data.next_pymnt_d],
            "last_credit_pull_d": [data.last_credit_pull_d],
            "collections_12_mths_ex_med": [data.collections_12_mths_ex_med],
            "application_type": [data.application_type],
            "acc_now_delinq": [data.acc_now_delinq],
            "tot_coll_amt": [data.tot_coll_amt],
            "tot_cur_bal": [data.tot_cur_bal],
            "total_rev_hi_lim": [data.total_rev_hi_lim]
        }
        date_cols = ["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"]

        df_input = pd.DataFrame(input_data)
        for cols in date_cols:
            df_input[cols] = pd.to_datetime(df_input[cols], format="%Y-%m-%dT%H:%M:%S.%fZ").view(np.int64) / 10**9

        desc_vectorized = tfidf_desc.transform(cvec_desc.transform([data.desc if data.desc is not None else ""]))
        emp_vectorized = tfidf_emp.transform(cvec_emp.transform([data.emp_title if data.emp_title is not None else ""]))

        data_transformed = pipeline_preprocessor.transform(df_input)

        X_transformed = hstack([data_transformed, df_input[date_cols] ,desc_vectorized, emp_vectorized])

        X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=columns_transformed)
        X_selected = X_transformed_df[selected_columns]

        prediction = model.predict(X_selected)
        predicted_class = np.argmax(prediction, axis=1)
        
        predicted_label = class_labels[predicted_class[0]]
        print(predicted_label)
        return {"prediction": predicted_label}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
