# Rubix ML - Customer Churn Predictor
Detect unhappy customers with Naive Bayes and 19 categorical features.

- **Difficulty:** Easy
- **Training time:** Seconds

## Installation
Clone the project locally using [Composer](https://getcomposer.org/):
```sh
$ composer create-project rubix/churn
```

## Requirements
- [PHP](https://php.net) 7.4 or above

## Tutorial
In traditional programming, developers are responsible for coding the behavior of a program. Machine Learning is a paradigm shift from traditional programming because it allows the system itself to write and rewrite parts of its program through training and data. For this reason, you can think of Machine Learning as “programming with data.” Integrating ML into your projects is therefore a practice of merging logic written by developers with logic that was learned by a Machine Learning algorithm. Today, we’ll talk about how you can start integrating Machine Learning in your PHP projects using the open-source Rubix ML library. We’ll formulate the problem of customer churn, train a model to predict unhappy customers, and then use that model to identify the potentially unhappy customers within our database.

Let’s start by introducing the problem of customer churn i.e. the rate at which your customers will discontinue use of your product or service over a period of time. If we could detect which of our customers are most likely to stop using our product, then we could take action to try to repair the relationship before the customer has already decided to leave. But, how do we as developers encode the ruleset i.e. the “business logic” that determines what an unhappy customer looks like?

Let's imagine for this example that we are running a telecommunications business. One thing we could do is we could ask our customer service department what customers say about our service. We might learn that our customers who live in a certain region were more likely to complain of slow Internet speed. We could then look for other Internet service customers in that area and assign a higher probability that they’ll cancel our service in the future. We might also learn that older customers were really happy with our streaming TV and movie selection and were therefore more likely to hold onto their subscription. We *could* start by encoding these rules out by hand, but this quickly becomes overwhelming when we consider all the different factors that contribute to overall customer satisfaction. Instead, we can feed examples of both satisfied and dissatisfied customers into a learning algorithm and have it learn the rules automatically. Then, we can take the model we trained and use it to make predictions about the customers in our database.

Before we begin training our model, we need to gather up all the samples of both satisfied and dissatisfied customers that we know about and label them accordingly. Then we'll determine which features of a customer aid in determining whether or not a customer becomes dissatisfied. For example, features such as age or how many times the customer called for tech support are probably good features to include in the dataset, but features such as eye color and whether or not the customer has a back yard are probably just noise in this case and may be counterproductive to include them. In the example below, we'll load the samples from the provided example dataset using the CSV extractor and then choose a subset of the features using the ColumnPicker. In Rubix ML, Extractors are iterators that stream data from storage into memory and can be wrapped by other iterators to modify the data in-flight. Note that we've included the label for each sample as the last column of the data table.


```php
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;

$extractor = new ColumnPicker(new CSV('dataset.csv', true), [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MonthsInService', 'Phone',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'TV', 'Movies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Region', 'Churn',
]);
```

In Rubix ML, dataset objects provide a high-level API that allows you to process the data and create subsets among other things. Next, we'll instantiate a Labeled dataset object by passing the extractor to the static `fromIterator()` method. This process reads the samples from our dataset into memory.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($extractor);
```

The next thing we'll do is create two subsets of the dataset to be used for training and testing purposes. The training set will be used by Naive Bayes during training to learn a model while the testing set will be used to gauge the model's accuracy after training. In the example below, we'll put 80% of the labeled samples into the training set and use the remaining 20% for validation later. We use different samples to train the model than to validate it because we want to test the learner on samples it has never seen before.

```php
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

The learning algorithm that we’re going to showcase today is a classifier called Naive Bayes. It is a relatively simple algorithm that uses counting and Bayes' Theorem to derive the conditional probabilities of an outcome given a sample consisting of categorical features. The term “naive” is in reference to the algorithm’s feature independence assumption. It's naive because, in the real world, most features have some interactions. In practice however, this assumption turns out not to be a big problem and is actually a benefit in some cases.

We can instantiate our Naive Bayes estimator with a set of parameters (called "hyper-parameters") that will control how the learner behaves. The current implementation of Naive Bayes has two hyper-parameters that we should be aware of. The `priors` parameter allows the user to specify the class prior probabilities (i.e. the probability that a particular class will be the outcome if chosen randomly) instead of the default which is to calculate the prior probabilities from the distribution in the training set. For example, if we know that our average churn rate is about 5% in real life, then we can specify that as the `"Yes"` class's prior probability and Naive Bayes will adapt accordingly. The second hyper-parameter is the smoothing parameter which controls the amount of Laplacian smoothing added to the conditional probabilities of each feature calculated during training. Smoothing is a form of regularization that prevents the model from being too "sure of itself" especially when the number of training samples is low. For the purposes of this example, we'll leave the `smoothing` parameter as the default value of 1.0. Feel free to play around with these settings to see how they effect the accuracy of the trained model.

```php
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes([
    "Yes" => 0.05,
    "No" => 0.95,
]);
```

In the example dataset, `MonthsInService`, `MonthlyCharges`, and `TotalCharges` feature columns have numerical values. Since all values in CSV format are interpreted as strings by default, we'll apply a preprocessing step that converts the numeric strings (ex. "42") in the dataset to their integer and floating point representations. For this, we'll use a handy Transformer called Numeric String Converter applied to the entire dataset in one go. In addition, since Naive Bayes is only compatible with categorical features, we'll also apply an Interval Discretizer set to derive 3 discrete categories or "levels" from the aforementioned continuous features in the dataset. In the context of the `MonthsInService` feature, you can think of this transformation as converting the number of months the customer has been in service to one of three discrete categories - "short", "medium", or "long." We'll wrap our Naive Bayes estimator in the Pipeline meta-Estimator and add both Transformers to the list of preprocessing steps to apply to the dataset. This way, datasets will automatically be preprocessed before training and inference. In addition, by wrapping both the dataset transformers and the estimator we can save both the transformer fittings as well as the model parameters as one object.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\IntervalDiscretizer;

$estimator = new Pipeline([
    new NumericStringConverter(),
    new IntervalDiscretizer(3),
], $estimator);
```

To train the model and fit the transformers, pass the training dataset object to the newly instantiated Pipeline  meta-Estimator by calling the `train()` method. Under the hood, the dataset is transformed and then the learning algorithm builds a model with parameters consisting of the conditional probabilities of each possible class outcome given a particular feature.

```php
$estimator->train($training);
```

We can verify that the estimator has been trained by calling the `trained()` method on the Estimator interface.

```php
var_dump($estimator->trained());
```

```sh
bool(true)
```

To return the set of predictions, pass the testing dataset object to the `predict()` method on the estimator after it has been trained. The class predictions are determined by outputting the class that returns the highest probability when presented with a particular testing sample.

```php
$predictions = $estimator->predict($testing);
```

We can view the class predictions by outputting them to the terminal like in the example below.

```php
print_r($predictions);
```

```sh
Array
(
    [0] => No
    [1] => No
    [2] => No
    [3] => Yes
    [4] => No
)
```

With the test predictions and their ground-truth labels in hand, we can now turn our focus to testing the model using the "holdout" method described above. We're looking for a model that can generalize its training to new samples. The method we use to determine generalization performance is called cross-validation and the holdout technique is one of the most straightforward approaches. The holdout technique is named as such because a certain percentage of the dataset is "held out" for testing purposes. The upside to this method is that it's quick and only requires training one model to produce a meaningful validation score. However, the downside to this technique is that, since the validation score for the model is only calculated from a portion of the samples, it has less coverage than methods that train multiple models and test them on different samples each time. The Rubix ML library provides an entire subsystem dedicated to cross-validation. In the next example, we're going to generate and save a JSON report that contains detailed metrics for us to measure the accuracy of the model.

Next we're going to instantiate a Multiclass Breakdown and Confusion Matrix report generator and wrap them in an Aggregate Report so they can be generated at the same time. Multiclass Breakdown is a detailed report containing scores for a multitude of metrics including Accuracy, Precision, Recall, F-1 Score, and more on an overall and per-class basis. Confusion Matrix is a table that pairs the predictions on one axis and their ground-truth on the other. Counting each pair gives us a sense for which classes the estimator might be "confusing" another class for.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$reportGenerator = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);
```

To create the report object call the `generate()` method on the report generator with the predictions and ground-truth labels from the testing set as arguments.

```php

$report = $reportGenerator->generate($predictions, $testing->labels());
```

Since report objects implement the Stringable interface, we can output the report by echoing it out to the terminal.

```php
echo $report
```

```json
[
    {
        "overall": {
            "accuracy": 0.7806955287437899,
            "balanced accuracy": 0.7405835852127411,
            "f1 score": 0.7301102604109521,
            "precision": 0.7226865136298422,
            "recall": 0.7405835852127411,
            "specificity": 0.7405835852127411,
            "negative predictive value": 0.7226865136298422,
            "false discovery rate": 0.27731348637015785,
            "miss rate": 0.2594164147872588,
            "fall out": 0.2594164147872588,
            "false omission rate": 0.27731348637015785,
            "mcc": 0.4629242695197278,
            "informedness": 0.4811671704254823,
            "markedness": 0.4453730272596843,
            "true positives": 1100,
            "true negatives": 1100,
            "false positives": 309,
            "false negatives": 309,
            "cardinality": 1409
        },
        "classes": {
            "No": {
                "accuracy": 0.7806955287437899,
                "balanced accuracy": 0.7405835852127411,
                "f1 score": 0.8469539375928679,
                "precision": 0.8689024390243902,
                "recall": 0.8260869565217391,
                "specificity": 0.6550802139037433,
                "negative predictive value": 0.5764705882352941,
                "false discovery rate": 0.13109756097560976,
                "miss rate": 0.17391304347826086,
                "fall out": 0.34491978609625673,
                "false omission rate": 0.42352941176470593,
                "informedness": 0.4811671704254823,
                "markedness": 0.4453730272596843,
                "mcc": 0.4629242695197278,
                "true positives": 855,
                "true negatives": 245,
                "false positives": 129,
                "false negatives": 180,
                "cardinality": 1035,
                "proportion": 0.7345635202271115
            },
            "Yes": {
                "accuracy": 0.7806955287437899,
                "balanced accuracy": 0.7405835852127411,
                "f1 score": 0.6132665832290363,
                "precision": 0.5764705882352941,
                "recall": 0.6550802139037433,
                "specificity": 0.8260869565217391,
                "negative predictive value": 0.8689024390243902,
                "false discovery rate": 0.42352941176470593,
                "miss rate": 0.34491978609625673,
                "fall out": 0.17391304347826086,
                "false omission rate": 0.13109756097560976,
                "informedness": 0.4811671704254823,
                "markedness": 0.4453730272596843,
                "mcc": 0.4629242695197278,
                "true positives": 245,
                "true negatives": 855,
                "false positives": 180,
                "false negatives": 129,
                "cardinality": 374,
                "proportion": 0.2654364797728886
            }
        }
    },
    {
        "No": {
            "No": 855,
            "Yes": 129
        },
        "Yes": {
            "No": 180,
            "Yes": 245
        }
    }
]
```

To save the report we can call the `saveTo()` method on the Encoding object that is returned by calling the `toJSON()` method on the Report object. In this example, we'll use the Filesystem Persister to save the report to a file named `report.json`.

```php
use Rubix\ML\Persisters\Filesystem;

$report->toJSON()->saveTo(new Filesystem('report.json'));
```

We'll also save the entire Pipeline so that we can use it within the context of our e-commerce system to identify potentially unhappy customers in our database. Rubix ML provides another meta-Estimator called Persistent Model that wraps a Persistable estimator and provides methods for saving and loading the model data from storage. In the example below we'll wrap our Pipeline object with Persistent Model and save it to the filesystem using the default RBX serializer.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel($estimator, new Filesystem('model.rbx'));

$estimator->save();
```

In practice, we'd probably spend more time iterating over the training and cross-validation process in an effort to fine-tune the dataset and hyper-parameters. We might also try out different classifiers such as Classification Tree or Logit Boost to see if they are better suited to our problem. For the next part of this tutorial, we're going to assume that we're fine with the model we've trained so far and we're ready to put it into production.

The next thing we need to determine is when to make predictions about our customers. For this problem, it makes a lot of sense to generate predictions for all our customers at the same time and store the value in the database alongside the customer's data. Then, we could periodically predict the new customers and update the existing customers using a script that runs in the background of our application. The nice thing about this design is that we don't need to keep the model loaded into memory. However, if you need the prediction for new customers instantly or if you have a quickly evolving model, you may want to consider doing inference on-the-fly. See the [Server](https://github.com/RubixML/Server) package for an example of how to do this in a performant way using asynchronous PHP.

We're going to start a new script now for predicting the churn label of the customers in our database. For demonstration, we've provided an example Sqlite database with over 2000 customers. Let's load some customers from the database and use our stored model to predict their likelihood of churn. First, instantiate a SqlTable extractor by passing a PDO connection object pointed to our Sqlite database.

```php
use Rubix\ML\Extractors\SqlTable;
use PDO;

$connection = new PDO('sqlite:database.sqlite');

$extractor = new SqlTable($connection, 'customers');
```

If we didn't want to load all the customers from our database all at once, we can use the PHP Limit Iterator to specify an offset and a limit like in the example below.

```php
$extractor = new LimitIterator($extractor->getIterator(), 0, 100);
```

Since we don't know the label of the customers in our database yet, we'll select the features of the samples without their corresponding labels. Make sure to load the features in the same order as they were given to train the model.

```php
$extractor = new ColumnPicker($extractor, [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MonthsInService', 'Phone',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'TV', 'Movies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Region',
]);
```

Now, instantiate an Unlabeled dataset by calling the `fromIterator()` factory method with the extractor.

```php
$dataset = Unlabeled::fromIterator($extractor);
```

With the samples loaded into memory, lets load the Pipeline we saved earlier into memory by calling the `load()` method on the Persistent Model meta-Estimator with a Filesystem persister pointing to the path of the model file in storage.

```php
$estimator = PersistentModel::load(new Filesystem('model.rbx'));
```

Finally, return the predictions for the customers in our database by passing the inference set to the `predict()` method on the Pipeline meta-Estimator and store the values in the database under the Churn column.

```php
$predictions = $estimator->predict($dataset);
```

## Original Dataset
https://github.com/codebrain001/customer-churn-prediction

## License
The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).