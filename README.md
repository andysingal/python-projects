# python-projects


# Writing and Maintaining Production ML Code
Many data scientists like the usability and interactivity of Jupyter Notebooks when they develop and evaluate models. It is convenient indeed to manipulate some code and immediately see a visual table or a chart, and most ML tutorials, examples, and Kagle projects are consumed as Notebooks.

You can find projects where the data preparation, training, evaluation, and even prediction are all made in one huge Notebook, but this approach can lead to challenges when moving to production, for example:

1. Very hard to track the code changes across versions (in Git).

2. Almost impossible to implement test harnesses and unit testing.

3. Functions cannot be reused in various projects.

4. Moving to production requires code refactoring and removal of visualization or scratch code.

5. Lack of proper documentation.

The best approach is to use functional programming for code segments and Notebooks for interactive and visualization parts. Example 2-1 implements a data preparation function that accepts a dataset (DataFrame) and some properties as inputs and returns the manipulated dataset. The function is documented and allows users to understand the purpose and usage.
```
import pandas as pd

def add_date_features(
    data, time_column: str = "timestamp", drop_timestamp: bool = False
):
    """Add numeric date features (day of week, hour, month) to a dataframe

    :param time_column:    The name of the timestamps column in the data
    :param drop_timestamp: set to True to drop the timestamp column from
                           the original dataframe
    :return datafarame
    """
    timestamp = pd.to_datetime(data[time_column])
    data["day_of_week"] = timestamp.dt.day_of_week
    data["hour"] = timestamp.dt.hour
    data["month"] = timestamp.dt.month
    if drop_timestamp:
        data.drop([time_column], axis=1, inplace=True)
    return data
```
Place the function in a separate Python file data_prep.py, and you can call it from the Notebook, inject data, and examine or visualize its output using the following code cell:

```
import pandas as pd
from data_prep import add_date_features

df = pd.read_csv("data.csv")
df = add_date_features(df, "timestamp", drop_timestamp=True)
df.head()
```
 Data prep test function (test_data_prep.py)
 ```
import pytest
import data_prep
import pandas as pd

# tell pytest to test both drop values (True/False)
@pytest.mark.parametrize("drop_timestamp", [True, False])
def test_add_date_features(drop_timestamp):
    df = pd.DataFrame({'times':['2022-01-01 08:00',
                                '2022-02-02 09:00',
                                '2022-03-03 10:00'],
                       'vals':[1,2,3]})
    new_df = data_prep.add_date_features(df, "times", drop_timestamp=drop_timestamp)

    # verify the results are as expected
    assert new_df["day_of_week"].to_list() == [5, 2, 3]
    assert new_df["month"].to_list() == [1, 2, 3]
    assert new_df["hour"].to_list() == [8, 9, 10]
    assert ("times" in new_df.columns.values) != drop_timestamp
```

Using this approach, you gain some immediate benefits:

- Easily see changes to your data prep code in the version control.

- The same code can be tested later with a test harness (for example, using pytest).

- The function can be moved to production without the need to refactor the notebook.

- The function is documented, and you can easily understand how to use it and what to expect.

- The function can later be saved to a shared library and used across different projects.

The code becomes more readable.

## Tracking and Comparing Experiment Results
When running ML experiments, it is essential to track every run so that you can reproduce experiment results (for example, which parameters and inputs yield the best results), visualize the various metrics, and compare the results of different algorithms or parameters sets.

<img width="739" alt="Screenshot 2023-11-03 at 9 15 45 PM" src="https://github.com/andysingal/python-projects/assets/20493493/a5d16588-f273-4679-8c31-abe6b4745395">

In the real world, experiments can run in an automated ML pipeline (see Figure 2-9), which comprises different steps (data prep, train, test, and so on). Each stage of the pipeline accepts parameters, inputs data, and generates results such as output values, metrics, and data to be used in subsequent pipeline steps. In addition, the tracking should be extended to operational data (which code was used, packages, allocated and used resources, systems, and so on).

<img width="691" alt="Screenshot 2023-11-03 at 9 16 35 PM" src="https://github.com/andysingal/python-projects/assets/20493493/1a798170-dce0-4d0d-a948-f952b171b081">

A good tracking system also records the code version, used packages, runtime environment and parameters, resources, code profiling, and so on

<img width="629" alt="Screenshot 2023-11-03 at 9 18 27 PM" src="https://github.com/andysingal/python-projects/assets/20493493/cbac2a09-d88e-4e98-8bf1-25f1242e6552">

Some MLOps frameworks provide auto-logging for ML/DL workloads where you can import a library that automatically records all the ML framework-specific metrics.

## Distributed Training and Hyperparameter Optimization 
To get to the best model results, try out various algorithms or parameter combinations and choose the best one based on a target metric like best accuracy. This work can be automated using multiple hyperparameter optimization and AutoML frameworks, which try out the different combinations, record all the metrics for each run and mark the best.

There are several hyperparameter execution strategies:

1. Grid search: Running all the parameter combinations

2. Random: Running a sampled set from all the parameter combinations

3. Bayesian optimization: Buiding a probability model of the objective function and using it to select the most promising hyperparameters to evaluate in the true objective function

4. List: Running the first parameter from each list followed by the second from each list and so on

## Building and Testing Models for Production
When models are used in real-world applications, it is critical to ensure they are robust and well-tested. Therefore, in addition to traditional software testing (unit tests, static tests, and so on), testing should cover the following categories:

1. Data quality tests: The dataset used for training is of high quality and does not carry bias.

2. Model performance tests: The model produces accurate results.

3. Serving application tests: The deployed model along with the data pre- or post-processing steps are robust and provide adequate performance.

4. Pipeline tests: Ensuring the automated development pipeline handles various exceptions and the desired scale.


Once you train the model, the next step is to make sure it is accurate and resilient. Beyond the common practice of setting aside a test dataset and measuring the model accuracy using that dataset, several additional tests can improve the model quality:

1. Verify the performance is maintained across essential slices of the data (for example, devices by model, users by country or other categories, movies by genre) and that it does not drop significantly for a specific group.

2. Compare the model results with previous versions or a baseline version and verify the performance does not degrade.

3. Test different parameter combinations (hyperparameter search) to verify we chose the best parameter combination.

4. Test for bias and fairness by verifying that the performance is maintained per gender and specific populations.

5. Check feature importances and whether there are features with a marginal contribution that can be removed from the model.

6. Test for immunity to fake, random, or malicious input vectors to increase robustness and defend against adversarial attacks.

In many cases, the trained model can be further optimized for production and higher performance, for example, by performing feature selection and removing redundant features or by compressing the models and storing them in more machine-efficient formats like ONNX. Therefore, ML pipelines may incorporate model optimization steps.

<img width="683" alt="Screenshot 2023-11-03 at 9 25 34 PM" src="https://github.com/andysingal/python-projects/assets/20493493/72446f88-5b00-459b-ab00-e9f786735726">

Resources:
- https://mlops.community/blog/
- https://geekflare.com/best-mlops-platforms/
- https://github.com/fugue-project/tune/blob/master/tune_notebook/monitors.py
- https://github.com/DataTalksClub/mlops-zoomcamp
- https://www.datacamp.com/blog/top-mlops-tools
- <strong>https://blog.csdn.net/m0_57236802/article/details/133696149</strong>
