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
In traditional programming, developers are responsible for coding the behavior of a program. Machine Learning is a paradigm shift from traditional programming because it allows the system itself to write and rewrite parts of its program through training and data. For this reason, you can think of Machine Learning as “programming with data.” Integrating ML into your projects is therefore a practice of merging logic written by developers with logic that was learned by a Machine Learning algorithm. Today, we’ll talk about how you can start integrating Machine Learning in your PHP projects using the open-source Rubix ML library. Specifically, we’ll formulate the problem of customer churn prediction, train a model to detect the unhappy customers in your database, and then integrate that model into a broader e-commerce system.

Let’s start by introducing the problem of predicting customer churn i.e. the rate at which your customers will discontinue use of your product over a period of time. If we could detect which of our customers are most likely to stop using our product, then we could take action to try to repair the relationship before the customer has already decided to leave. But, how do we as developers encode the ruleset i.e. the “business logic” that determines what an unhappy customer looks like?

Let's assume for this example that we are running an Internet service provider (ISP). One thing we could do is we could ask our customer service department what customers say about our service. We might learn that our customers who live in a certain area were more likely to complain of slow Internet speed. We could then look for other customers in that area and assign a higher probability that they’ll cancel our service in the future. We might also learn that older customers were really happy with our senior citizen discount and therefore will be less likely to cancel our service in the future. We *could* start by encoding these rules by hand, but this quickly becomes overwhelming when we consider all the different factors that contribute to overall customer satisfaction. Instead, we can feed examples of both satisfied and dissatisfied customers into a learning algorithm and have it learn the rules automatically. Then, we can take the model we trained and use it to make predictions about customers in our database.

Before we begin training our model, we need to gather up all the samples of both satisfied and dissatisfied customers that we know about and label them accordingly. Then we'll determine which features of a customer aid in determining whether or not a customer becomes dissatisfied. For example, features such as age or how many times the customer called for tech support are probably good features to include in the dataset, but features such as eye color and whether or not the customer has a back yard are probably just noise in this case and may be counterproductive to include them. In the example below, we'll load the samples from the provided example dataset using the CSV extractor and then choose a subset of the features using the ColumnPicker. In Rubix ML, Extractors are iterators that stream data from storage into memory and can be wrapped by other iterators to modify the data in-flight. Note that we've included the label for each sample as the last column of the data table.


```php
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;

$extractor = new ColumnPicker(new CSV('dataset.csv', true), [
    "Gender", "SeniorCitizen", "Partner", "Dependents", "MonthsInService", "Phone",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "TV", "Movies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]);
```

Next, we'll instantiate a Labeled dataset object using the `fromIterator()` method. In Rubix ML, dataset objects provide a high-level API that allows you to process the data and create subsets among other things.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($extractor);
```

The learning algorithm that we’re going to explore today is a classifier called Naive Bayes. It is a relatively simple algorithm that uses counting and Bayes' Theorem to derive the conditional probability of an outcome given a set of categorical features. The term “naive” is in reference to the algorithm’s feature independence assumption which avoids having to account for interactions between features. It's naive because, in the real world, most features have some interactions. However, in practice, this feature independence assumption turns out not to be a problem and actually a benefit in some cases.

In the example dataset, `MonthsInService`, `MonthlyCharges`, and `TotalCharges` feature columns have numerical values. Since all values in CSV format are interpreted as strings by default, we'll apply a preprocessing step that converts the numeric strings (ex. "42") in the dataset to their integer and floating point representations. For this, we'll use a  Transformer called Numeric String Converter applied to the entire dataset in one go. In addition, since Naive Bayes is only compatible with categorical type features, we'll also apply an Interval Discretizer which is set to derive 3 discrete categories or "levels" from the aforementioned continuous features in the dataset. In the context of the `MonthsInService` feature, you can think of this process as converting the number of months the customer has been in service to one of three discrete categories - "short", "medium", or "long."

```php
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\IntervalDiscretizer;

$dataset->apply(new NumericStringConverter())
    ->apply(new IntervalDiscretizer(3));
```

With our dataset already prepared, the next thing we'll do is create two subsets to be used for training and validation purposes. The training set will be used by Naive Bayes during training to learn a model while the testing set will be used to gauge the model's accuracy after training. In the example below, we'll put 80% of the labeled samples into the training set and use the remaining 20% for validation later. We use different samples to train the model than to validate it because we want to test the learner on samples it has never seen before.

```php
[$training, $testing] = $dataset->stratifiedSplit(0.8);
```

Now, we can instantiate our Naive Bayes estimator with a set of parameters (called "hyper-parameters") that will control how the learner behaves. The current implementation of Naive Bayes has two hyper-parameters that we should be aware of. The `priors` parameter allows the user to specify the class prior probabilities (i.e. the probability that a particular class will be the outcome if chosen randomly) instead of the default which is to calculate the prior probabilities using the distribution of the training set. For example, if we know that our average churn rate is about 5% in real life, then we can specify that as the `"Yes"` class prior probability and Naive Bayes will adapt accordingly. The second hyper-parameter is the smoothing parameter which controls the amount of Laplacian smoothing added to the conditional probabilities of each feature calculated during training. Smoothing is a form of regularization that prevents the model from being too "sure of itself" especially when the number of training samples is low. For the purposes of this example, we'll leave the `smoothing` parameter as the default value of 1.0. Feel free to play around with these settings to see how they effect the accuracy of the trained model.

```php
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes([
    "Yes" => 0.05,
    "No" => 0.95,
]);
```

To train the model, pass the training dataset object to the newly instantiated Naive Bayes estimator by calling the `train()` method. Under the hood, the learning algorithm builds a model with parameters consisting of the conditional probabilities of each possible class outcome given a particular feature. Since we're dealing with discrete (categorical) features and outcomes, the algorithm calculates these probabilities by counting the number of instances of each category and then dividing over the total number of samples to obtain the probability density. 

```php
$estimator->train($training);
```

Then, to return a set of predictions from the testing set, pass it to the `predict()` method on the estimator after it has been trained. The class predictions are determined by taking the class that returns the highest probability when presented with a particular testing sample. The class probabilities are calculated by combining each of the conditional probabilities for a sample with the class prior probabilities using Bayes' Theorem.

```php
$predictions = $estimator->predict($testing);
```

With the test predictions and their ground-truth labels in hand, we can now turn our attention to measuring the accuracy of the model using the "holdout" method described above. We're looking for a model that can generalize its training to new samples. The method we use to determine generalization performance is called cross-validation and the holdout method is one of the most straightforward approaches. The holdout method is named as such because a certain percentage of the dataset is "held out" for testing purposes. The upside to this method is that it's quick to perform and only requires one model to be trained to produce a meaningful validation score. However, the downside to this technique is that, since the validation score for the model is only computed using a portion of the samples, it has less coverage than methods that train multiple models and test them on different samples each time.

The Rubix ML library provides an entire subsystem dedicated to cross-validation. In this example, we're going to score the model using a simple Accuracy metric and then generate and save a JSON report that has more detailed metrics. First, instantiate the Accuracy metric by calling its constructor. Then, pass the array containing the test predictions along with their ground-truth labels to the `score()` method. The return value is a scalar that we'll output to the terminal right after it's calculated.

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo $score;
```

```sh
0.77998580553584
```

Next, we're going to generate a couple reports using the same test predictions and their corresponding ground-truth labels. In this case, we'll instantiate both Multiclass Breakdown and Confusion Matrix report generators and wrap them in an Aggregate Report so they can be generated at the same time. Multiclass Breakdown is a detailed report containing scores for a multitude of metrics including Accuracy, Precision, Recall, F-1 Score, and more on both an overall and per-class basis. Confusion Matrix is a table that pairs the predictions on one axis and their ground-truth on the other. Counting each pair gives us a sense for which classes the estimator might be "confusing" another class for.


```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$reportGenerator = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$report = $reportGenerator->generate($predictions, $testing->labels());
```

```json
[
    {
        "overall": {
            "accuracy": 0.7799858055358411,
            "balanced accuracy": 0.73583146038389,
            "f1 score": 0.7273674880782962,
            "precision": 0.7209892323185374,
            "recall": 0.73583146038389,
            "specificity": 0.73583146038389,
            "negative predictive value": 0.7209892323185374,
            "false discovery rate": 0.27901076768146255,
            "miss rate": 0.2641685396161099,
            "fall out": 0.2641685396161099,
            "false omission rate": 0.27901076768146255,
            "mcc": 0.4565795150323564,
            "informedness": 0.47166292076778005,
            "markedness": 0.4419784646370748,
            "true positives": 1099,
            "true negatives": 1099,
            "false positives": 310,
            "false negatives": 310,
            "cardinality": 1409
        },
        "classes": {
            "No": {
                "accuracy": 0.7799858055358411,
                "balanced accuracy": 0.73583146038389,
                "f1 score": 0.8471400394477318,
                "precision": 0.865055387713998,
                "recall": 0.8299516908212561,
                "specificity": 0.6417112299465241,
                "negative predictive value": 0.5769230769230769,
                "false discovery rate": 0.13494461228600196,
                "miss rate": 0.17004830917874392,
                "fall out": 0.3582887700534759,
                "false omission rate": 0.42307692307692313,
                "informedness": 0.47166292076778005,
                "markedness": 0.4419784646370748,
                "mcc": 0.4565795150323564,
                "true positives": 859,
                "true negatives": 240,
                "false positives": 134,
                "false negatives": 176,
                "cardinality": 1035,
                "proportion": 0.7345635202271115
            },
            "Yes": {
                "accuracy": 0.7799858055358411,
                "balanced accuracy": 0.73583146038389,
                "f1 score": 0.6075949367088607,
                "precision": 0.5769230769230769,
                "recall": 0.6417112299465241,
                "specificity": 0.8299516908212561,
                "negative predictive value": 0.865055387713998,
                "false discovery rate": 0.42307692307692313,
                "miss rate": 0.3582887700534759,
                "fall out": 0.17004830917874392,
                "false omission rate": 0.13494461228600196,
                "informedness": 0.47166292076778005,
                "markedness": 0.4419784646370748,
                "mcc": 0.4565795150323564,
                "true positives": 240,
                "true negatives": 859,
                "false positives": 176,
                "false negatives": 134,
                "cardinality": 374,
                "proportion": 0.2654364797728886
            }
        }
    },
    {
        "No": {
            "No": 859,
            "Yes": 134
        },
        "Yes": {
            "No": 176,
            "Yes": 240
        }
    }
]
```

```php
use Rubix\ML\Persisters\Filesystem;

$report->toJSON()->saveTo(new Filesystem('report.json'));
```

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel($estimator, new Filesystem('model.rbx'));

$estimator->save();
```

## Original Dataset
https://github.com/codebrain001/customer-churn-prediction

## License
The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).