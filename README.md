# Supply Chain Optimization and Analysis
Project Overview
This project focuses on analyzing and optimizing supply chain operations using a dataset that includes information about product types, sales, stock levels, and other relevant factors. The primary objectives are to:

Analyze Data: Understand the relationships between various supply chain factors and their impact on revenue.
Build Predictive Models: Use machine learning techniques to predict revenue based on input features.
Optimize Costs: Apply linear programming to minimize costs under given constraints.
Visualize Results: Create an interactive dashboard to visualize revenue and other key metrics.
Dataset
The dataset used for this project is a supply chain dataset with the following columns:

Product type: Type of product (e.g., cosmetics, haircare, skincare)
SKU: Stock Keeping Unit identifier
Price: Price of the product
Availability: Availability status
Number of products sold: Total products sold
Revenue generated: Revenue generated from sales
Customer demographics: Demographics of customers
Stock levels: Current stock levels
Lead times: Time taken from order to delivery
Order quantities: Number of products ordered
Shipping times: Time taken for shipping
Shipping carriers: Shipping carriers used
Shipping costs: Costs associated with shipping
Supplier name: Name of the supplier
Location: Location of the supplier
Lead time: Time required for production
Production volumes: Volume of products produced
Manufacturing lead time: Time taken for manufacturing
Manufacturing costs: Costs of manufacturing
Inspection results: Results of product inspections
Defect rates: Rates of defects in products
Transportation modes: Modes of transportation used
Routes: Shipping routes taken
Costs: Overall costs
Methods
Data Preprocessing
Handling Missing Values: Filled missing values in numeric columns with the mean of each column.
Encoding Categorical Variables: Used LabelEncoder to convert categorical columns into numeric values for analysis.
Exploratory Data Analysis (EDA)
Correlation Heatmap: Generated a heatmap to visualize the correlation between numeric features, helping identify significant relationships.

python
Copy code
# Compute correlation matrix
correlation = data[numeric_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
Machine Learning Model
Feature and Target Definition: Defined features and target variable (Revenue generated).

Model Training: Used RandomForestRegressor to predict revenue.

Model Evaluation: Assessed model performance using Mean Squared Error (MSE).

python
Copy code
# Define features and target
features = data.drop(['Revenue generated'], axis=1)
target = data['Revenue generated']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
Cost Optimization
Linear Programming: Applied linear programming to minimize costs under constraints.

python
Copy code
from scipy.optimize import linprog

# Example cost coefficients and constraints
c = [data['Costs'].mean()]  # Cost per unit
A = [[1]]  # Constraints matrix (example)
b = [data['Stock levels'].mean()]  # Constraint value (example)

# Solve the optimization problem
result = linprog(c, A_ub=A, b_ub=b, method='simplex')

print("Optimal Order Quantity:", result.x)
print("Minimum Cost:", result.fun)
Visualization
Interactive Dashboard: Built a Dash application to create an interactive dashboard that visualizes revenue by product type.

python
Copy code
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Initialize the app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Supply Chain Optimization Dashboard"),
    dcc.Graph(id='revenue-graph'),
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
])

@app.callback(Output('revenue-graph', 'figure'), [Input('interval-component', 'n_intervals')])
def update_graph(n):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Product type'], y=data['Revenue generated'], name='Revenue by Product Type'))
    fig.update_layout(title='Revenue by Product Type', xaxis_title='Product Type', yaxis_title='Revenue')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
Key Findings and Results
Correlation Insights: The heatmap revealed strong correlations between certain features, such as Price and Revenue generated.
Model Performance: The Random Forest model provided insights into factors affecting revenue, with an evaluated MSE of [9550764.101039082].
Cost Optimization: The linear programming solution provided an optimal order quantity of [0.] with a minimum cost of [0.0].
Dashboard Visualization: The interactive dashboard allows users to explore revenue trends by product type, offering actionable insights for decision-making.
Conclusion
This project demonstrates the use of data analysis, machine learning, and optimization techniques to enhance supply chain operations. The interactive dashboard adds a layer of usability, enabling stakeholders to visualize key metrics and make informed decisions.
