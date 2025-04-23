import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

csv_file = 'Bigmart.csv'
target = 'OutletSales'
rf_pipeline = None
def predict_sales(csv_file, target = None):
    bigmart_data = pd.read_csv(csv_file)
    bigmart_data = bigmart_data.head(1000)

    # Ensure the taget feature exists in the DataFrame
    if target not in bigmart_data.columns:
        raise ValueError(f"Feature '{target}' not found in the DataFrame.")

    # Assuming 'OutletSales' is the target variable and other columns are features
    X = bigmart_data.drop(target, axis=1)
    y = bigmart_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(), ['Weight', 'ProductVisibility', 'MRP', 'EstablishmentYear']), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['ProductID','FatContent', 'ProductType', 'OutletID', 'OutletSize', 'LocationType', 'OutletType'])
        ])

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    # save model
    model_file = 'model.pkl'
    joblib.dump(rf_pipeline, model_file)

    return model_file, mse

model_file, mse_bigmart = predict_sales(csv_file, target)
print("Mean Squared Error:", mse_bigmart)