import joblib
import numpy as np

from src.logger_config import setup_logging

logger = setup_logging()

data = joblib.load("MLmodels/car_price_package.joblib")
model = data["model"]
brand_mapping = data["mapping"]


def get_predicted_price(year, manufacturer):
    brand = manufacturer.lower().strip()

    # Manufacturer validation
    if brand not in brand_mapping:
        logger.warning(f"Brand '{manufacturer}' not found in the database.")
        return None

    # Data transformation
    brand_id = brand_mapping[brand]
    input_data = np.array([[year, brand_id]])

    # Perform prediction
    prediction = model.predict(input_data)[0]

    # Safety check: ensure the price is not negative
    final_price = max(0, prediction)

    logger.info(
        f"Predicted result for {brand.capitalize()} ({year}): ${final_price:,.2f}"
    )
    return final_price


# Test run when script is executed directly
if __name__ == "__main__":
    get_predicted_price(2338, "ford")
