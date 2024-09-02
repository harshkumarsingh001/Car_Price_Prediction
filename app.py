from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

def map_brand_to_numeric(brand):
    brands = ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep',
              'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo',
              'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel']
    return brands.index(brand) + 1 if brand in brands else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        name = request.form.get('name')
        year = int(request.form.get('year'))
        km_driven = int(request.form.get('km_driven'))
        fuel = request.form.get('fuel')
        seller_type = request.form.get('seller_type')
        transmission = request.form.get('transmission')
        owner = request.form.get('owner')
        mileage = float(request.form.get('mileage'))
        engine = int(request.form.get('engine'))
        max_power = float(request.form.get('max_power'))
        seats = int(request.form.get('seats'))

        # Prepare input data for prediction
        input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
                                  columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

        # Data transformation to match model's expectations
        input_data['owner'] = input_data['owner'].replace(['first_owner', 'second_owner', 'Third Owner', 'fourth_owner', 'test_drive_car'], [1, 2, 3, 4, 5])
        input_data['fuel'] = input_data['fuel'].replace(['Petrol', 'Diesel', 'CNG', 'lpg', 'electric'], [1, 2, 3, 4, 5])
        input_data['seller_type'] = input_data['seller_type'].replace(['individual', 'Dealer', 'trustmark_dealer'], [1, 2, 3])
        input_data['transmission'] = input_data['transmission'].replace(['manual', 'Automatic'], [1, 2])
        input_data['name'] = input_data['name'].apply(map_brand_to_numeric)

        # Predict the car price
        car_price = model.predict(input_data)

        # Return the result to the result.html template
        return render_template('result.html', car_price=round(car_price[0], 2))
    except Exception as e:
        # Return error message if any exception occurs
        return render_template('result.html', car_price=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
