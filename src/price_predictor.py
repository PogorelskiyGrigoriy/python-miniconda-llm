import joblib
import numpy as np
import logging

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the package once upon module import
try:
    data = joblib.load('MLmodels/car_price_package.joblib')
    model = data['model']
    brand_mapping = data['mapping']
except FileNotFoundError:
    logger.error("Model file not found! Please run the training script first.")

def get_predicted_price(year, manufacturer):
    """
    Takes year and manufacturer string, returns the predicted price.
    """
    brand = manufacturer.lower().strip()
    
    # Manufacturer validation
    if brand not in brand_mapping:
        available = ", ".join(list(brand_mapping.keys())[:10]) # Show first 10 examples
        logger.warning(f"Brand '{manufacturer}' not found in the database.")
        logger.info(f"Available brands (examples): {available}...")
        return None
    
    # Data transformation
    brand_id = brand_mapping[brand]
    input_data = np.array([[year, brand_id]])
    
    # Perform prediction
    prediction = model.predict(input_data)[0]
    
    # Safety check: ensure the price is not negative
    final_price = max(0, prediction)
    
    logger.info(f"Predicted result for {brand.capitalize()} ({year}): ${final_price:,.2f}")
    return final_price

# Test run when script is executed directly
if __name__ == "__main__":
    get_predicted_price(2028, "toyota")