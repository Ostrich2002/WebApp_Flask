

# this works completely fine
#when we upload our bostonhousting dataset then the documents in our mongoDB database increases so final output keep increasing everytime by approx 500 values as this dataset has 500 rows 


from flask import Flask, render_template, request
from pymongo import MongoClient
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA #added later for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['data_analysis_db']


@app.route('/', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            dataset = pd.read_csv(file)
            collection = db['dataset_collection']
            collection.insert_many(dataset.to_dict('records'))
            return 'Dataset uploaded successfully!'
    return render_template('index.html')


@app.route('/analysis', methods=['GET', 'POST'])
def perform_analysis():
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        collection = db['dataset_collection']
        dataset = pd.DataFrame(list(collection.find()))

        if algorithm == 'linear_regression':
            
            # Exclude 'ObjectId' field from the dataset
            dataset = dataset.drop('_id', axis=1) #extra line of code to remove error

            X = dataset.drop('price', axis=1)
            y = dataset['price']

            # Preprocess the data  ----these two lines are extra
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            # Preprocess the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X_scaled, y)

            # Perform prediction on the dataset
            predictions = model.predict(X_scaled)

            # Store the results in a variable
            result = predictions.tolist()

            return render_template('result.html', result=result)
        
        #code below this was added later on to implement PCA
        else:
            if algorithm == 'pcaPCA':
            # Exclude '_id' field from the dataset
                dataset = dataset.drop('_id', axis=1)

                X = dataset.drop('price', axis=1)  # Replace 'target_variable' with your target column name

                # Preprocess the data
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)

                # Preprocess the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_imputed)

                # Perform PCA
                pca = PCA()
                principal_components = pca.fit_transform(X_scaled)

                # Get the explained variance ratio
                explained_variance_ratio = pca.explained_variance_ratio_

                # Store the results in variables
                principal_components_list = principal_components.tolist()
                explained_variance_ratio_list = explained_variance_ratio.tolist()

                return render_template('pca_result.html', principal_components=principal_components_list,
                                    explained_variance_ratio=explained_variance_ratio_list)
                #upto here the code was inserted leter on for PCA

    return render_template('analysis.html')


if __name__ == '__main__':
    app.run(debug=True)


