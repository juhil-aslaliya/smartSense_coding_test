<h1>Instructions</h1>

- Install the required packages as mentioned in `requirements.txt`
- Then run the src file in the terminal using the command: `python src.py`

<h1>Approach</h1>

- The brochures are collected from the official websites of Maruti Suzuki, Hyundai, Nissan, Honda, and Renault.
- Used multiclass logistic regression for each of the car categories in a one-vs-all classification

<h1>Explanation</h1>

- A car is categorized by its dimensions and the engine power.
- I have done the categorization based on the dimensions only, as considering the small size of the data, more features would have led to overfitting
- Upon plotting the variables, some relations between the features and the categories were apparent of a categorization based on a threshold.

<h1>Future Scope</h1>

- More data can be collected to make the model more robust and less prone to overfitting.
- Some other classification methods like decision trees could be used.
- Boosting and/or bagging can be added to the training, so that the model does not become biased towards fitting the majority class.
