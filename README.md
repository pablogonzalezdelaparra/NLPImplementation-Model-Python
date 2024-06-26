# NLP Model for Text Similarity Detection

This Natural Language Processing (NLP) model is implemented in Python and is designed to detect text similarity in abstracts. It utilizes various frameworks and techniques to analyze and compare text data.

## Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 201)
### Evidencia 1 - Fase 2 - Parte B: Implementación usando NLP
### Team Members:
- Aleny Sofía Arévalo Magdaleno - A01751272
- Pablo González de la Parra - A01745096
- Valeria Martínez Silva - A01752167

## Frameworks Used

The model is built using the following frameworks:

- Python - 3.12.3
- [unittest](https://docs.python.org/3/library/unittest.html) - Python's built-in testing framework
- [nltk](https://www.nltk.org/) - Natural Language Toolkit for NLP tasks
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning library for Python

## How to Run Tests

To run all the tests, execute the following command:

```python
python3 -m unittest discover -s tests
```

To run a specific test, execute the following command:

```python
python3 -m unittest tests.test_example
```

Replace `test_example` with the name of the specific test module you want to run.

## Getting Started

To get started with the NLP model:

1. Clone the repository:

   ```python
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```python
   cd NLPImplementation-Model-Python
   ```

3. Install the required dependencies:

   ```python
   pip3 install -r requirements.txt
   ```

4. Run the tests to ensure everything is set up correctly:

   ```python
   python3 -m unittest discover -s tests
   ```

5. Start using the NLP model in your own projects by importing and utilizing the `NLPModel` class.
   ```python
   # You can use this file to run the model with the data provided in the data folders

   python3 run_model.py
   ```

## Training and Testing Data

The training and testing data are stored in two different directories:

- `train_data` - Contains the training data for the NLP model
- `test_data` - Contains the testing data for the NLP model

The data is stored in plain text files, with each file containing a single abstract. The abstracts are used to train and test the model for text similarity detection.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code as needed.
