# Rubix ML - Customer Churn Predictor

Machine Learning is a paradigm shift from traditional programming because it allows the software itself to modify its programming through training and data. For this reason, you can think of Machine Learning as “programming with data.” Integrating ML into your project is therefore a practice of merging logic written by developers with logic that was learned by a Machine Learning algorithm. Today, we’ll talk about how you can start integrating Machine Learning models into your PHP projects using the open-source Rubix ML library. We’ll formulate the problem of customer churn prediction, train a model to identify what an unhappy customer looks like, and then use that model to detect the unhappy customers within our database.

## Installation

Clone the project locally using [Composer](https://getcomposer.org/):

```sh
$ composer create-project rubix/churn
```

## Requirements

- [PHP](https://php.net) 8.3 or above

## Tutorial

### Introduction

Let’s start by introducing the problem of predicting customer churn. Churn rate is the rate at which customers discontinue use of a product or service over a period of time. If we could predict which of our customers are most likely to leave, then we could take action to try to repair the relationship before they are gone. But, how do we as developers encode the ruleset i.e. the “business logic” that determines what an unhappy customer looks like?

Imagine that you are a developer working at a telecommunications company tasked with optimizing customer churn. One thing you could do is ask the customer service department what customers say about the service. You might learn that our customers who live in a certain region were more likely to complain of slow Internet speed and discontinue their service. You might also learn that older customers were really happy with the streaming TV and movie selection and were therefore more likely to hold onto their subscription. You *could* start by encoding these rules out by hand, but this quickly becomes overwhelming when you consider all the different factors that contribute to customer satisfaction. Instead, we can feed samples of both satisfied and unsatisfied customers through a learning algorithm and have the learner learn the rules automatically. Then, we can take that model and use it to make predictions about the customers in our database.

### Preparing the Dataset

Before training the model, we need to gather the samples of satisfied and unsatisfied customers and label them accordingly. Then, we'll determine which features of a customer are beneficial in determining whether or not a customer will churn. For example, service region and the number of times the customer called for tech support are probably good features to include in the dataset, but features such as eye color and whether or not the customer has a back yard or not may be counterproductive to include them. In the example below, we'll load the samples from the provided example dataset using the CSV extractor and then select a subset of the features using the ColumnPicker. In Rubix ML, Extractors are iterators that stream data from storage into memory and can be wrapped by other iterators to modify the data in-flight. Note that we've included the label for each sample as the last column of the data table as is the convention.

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

In Rubix ML, dataset objects provide a high-level API that allow you to process the samples and create subsets among other things. Next, we'll instantiate a Labeled dataset object by passing the extractor object to the static `fromIterator()` method.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($extractor);
```

### Preprocessing the Dataset

In the example dataset, `MonthsInService`, `MonthlyCharges`, and `TotalCharges` all have numerical values. Since all values in CSV format are interpreted as strings by default, we'll need to apply a preprocessing step that converts the numeric strings (ex. "42") in the dataset to their floating point representations. For this, we'll apply a stateless Transformer called [Float Type Converter](https://rubixml.github.io/ML/3.0/transformers/float-type-converter.html) to convert those values in the first preprocessing step. Logit Boost learns regression trees that split on numeric thresholds, so the numerical features can be left in their continuous form.

Because Float Type Converter is stateless i.e. it has no learned parameters, there is no fitted transformer to persist alongside the model. We can simply apply a fresh instance directly to the dataset by calling the `apply()` method. In this example, we apply the transformer to the entire labeled dataset before splitting it into subsets so that all samples share the same preprocessing.

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());
```

The next thing we'll do is create two subsets of the dataset to be used for training and testing. The training set will be used by Grid Search to learn a model and the testing set will be used to gauge the model's accuracy after training. Stratifying the samples by label ensures that the class proportions are maintained in both subsets. In the example below, we'll put 90% of the labeled samples into the training set and use the remaining 10% for validation later using the stratified splitting method.

> **Note:** The reason we use different samples to train the model than to validate it is because we want to test the learner on samples it has never seen before.

```php
[$training, $testing] = $dataset->stratifiedSplit(0.9);
```

Since we're going to validate the model from a separate script later on, we'll export both subsets to their own CSV files so that the exact same held-out samples can be reused without re-running training. The `exportTo()` method writes a dataset to storage using the provided Exporter - in this case the standard CSV exporter with `overwrite` enabled so that files from a previous run are replaced.

```php
use Rubix\ML\Extractors\CSV;

$training->exportTo(new CSV('training.csv'), overwrite: true);
$testing->exportTo(new CSV('testing.csv'), overwrite: true);
```

### Training the Model

Logit Boost is a stage-wise additive ensemble that uses regression trees to iteratively learn a logistic regression model for binary classification. Instead of learning the labels directly, each boosting round trains a small regression tree to follow the gradient of the cross entropy loss function - in other words, to predict the current ensemble's error. The tree's predictions are then added to the ensemble scaled by a small learning rate, and the process repeats, concentrating more and more effort on the samples the model is least certain about. Because the base learners are decision trees, the ensemble can capture non-linearities and interactions between features that a linear model would miss.

To instantiate our Logit Boost estimator we need to decide on a set of parameters (called "hyper-parameters") that will control how the learner behaves. The `booster` hyper-parameter is the base regressor used to fit the loss residuals - we'll use a [Regression Tree](https://rubixml.github.io/ML/3.0/regressors/regression-tree.html) so that each round contributes only a small, specialized tree to the ensemble. The `rate` is the learning rate i.e. the *shrinkage* applied to each step, keeping the influence of every booster modest so that the ensemble generalizes better. The `ratio` determines the proportion of training samples subsampled to train each booster, which injects a bit of randomness into the fitting process similar to bagging. The `epochs` hyper-parameter caps the number of boosting rounds at 1000, while `minChange` stops training early once the improvement in the cross entropy loss falls below `1e-5`. The remaining hyper-parameters (`evalInterval`, `window`, and `metric`) govern validation-based progress monitoring and early stopping, which require a validation set to be supplied with `setValidationDataset()` first.

Rather than committing to a single combination of those hyper-parameters up front, we'll let [Grid Search](https://rubixml.github.io/ML/3.0/grid-search.html) find a good one for us. Grid Search is a meta-estimator that trains one model for every combination of hyper-parameters in a grid, scores each combination using cross-validation, and then re-trains the winning combination on the full training dataset, exposing it as its base estimator. We'll construct the search with the `fromNamedParams()` factory by passing the class of the base learner along with the list of candidate values for each hyper-parameter we want to tune. Here we search over the depth of the `booster` tree (3 or 4), the `rate` (0.1 or 0.3), and the `ratio` (0.3, 0.5, or 0.7) - 12 combinations in total - while the remaining hyper-parameters are left at their default values. By default, each combination is scored by 5-fold cross-validation using the F Beta metric.

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\LogitBoost;
use Rubix\ML\Regressors\RegressionTree;

$estimator = GridSearch::fromNamedParams(
    class: LogitBoost::class,
    params: [
        'booster' => [new RegressionTree(3), new RegressionTree(4)],
        'rate' => [0.1, 0.3],
        'ratio' => [0.3, 0.5, 0.7],
    ],
);
```

Every trial in the search is independent of the others, so we can hand the work to a parallel processing backend to run them concurrently. The [Amp](https://rubixml.github.io/ML/3.0/backends/amp.html) backend executes the trials as asynchronous coroutines, cutting down the wall clock time of the search. We'll also attach a logger to the search so that it reports the score of each trial to the terminal as it completes.

```php
use Rubix\ML\Backends\Amp;
use Rubix\ML\Loggers\Screen;

$logger = new Screen();

$estimator->setBackend(new Amp());

$estimator->setLogger($logger);
```

As we mentioned earlier, Logit Boost's progress monitoring and early stopping need a validation dataset. Grid Search exposes a `setup()` method for exactly this kind of configuration - it registers a callback that is invoked on every candidate estimator before it is cross-validated, and again on the final winner before it is retrained on the full training set. Here we use it to hand each candidate the held-out testing set:

```php
$estimator->setup(function (LogitBoost $estimator) use ($testing) {
    $estimator->setValidationDataset($testing);
});
```

Now we're ready to train the model by passing the training dataset to the estimator. In one call, Grid Search cross-validates every combination, selects the best one, and trains a final model with it on the full training set.

```php
$estimator->train($training);
```

We can verify that the learner has been trained by calling the `trained()` method on the Estimator interface.

```php
var_dump($estimator->trained());
```

```sh
bool(true)
```

Once the search is finished, we can inspect how each trial scored by echoing out the Report returned from the `results()` method. The rows are keyed by trial number in the order the trials were trained in, and each row contains the hyper-parameters that were tested along with the cross-validation score they received.

```php
echo $estimator->results();
```

```json
{
    "Trial 1": {
        "booster": "Regression Tree (max height: 3, max leaf size: 5, max features: null, min purity increase: 1.0E-7, max bins: null)",
        "rate": "0.1",
        "ratio": "0.3",
        "epochs": "1000",
        "minChange": "1.0E-5",
        "evalInterval": "3",
        "window": "5",
        "metric": "null",
        "F Beta (beta: 1)": "0.73822930920074"
    },
    "Trial 2": {
        "booster": "Regression Tree (max height: 3, max leaf size: 5, max features: null, min purity increase: 1.0E-7, max bins: null)",
        "rate": "0.1",
        "ratio": "0.5",
        "epochs": "1000",
        "minChange": "1.0E-5",
        "evalInterval": "3",
        "window": "5",
        "metric": "null",
        "F Beta (beta: 1)": "0.73725539246956"
    },
    ...
}
```

You can see the same scores in the log lines emitted by the logger, one for each trial as it completes, followed by a line announcing the winning combination of hyper-parameters. In the example above, Trial 1 - a depth 3 booster with a `rate` of 0.1 and a `ratio` of 0.3 - came out on top with an F Beta of about 0.738, so that is the combination that was retrained on the full training set.

To better understand what happened when we called the `train()` method let's peak under the hood of the boosting process for a brief moment. The algorithm begins with an ensemble that is completely ignorant about the labels and then loops over the epochs. At each epoch it computes the gradient of the cross entropy loss with respect to the ensemble's current predictions, fits the `booster` regression tree to that gradient using a random subsample of the training set weighted by how badly the current model is doing, and adds the tree's predictions to the ensemble scaled by the `rate`. Samples that the model currently misclassifies receive the largest weights, so each successive round focuses harder on the customers that are most difficult to separate. Training continues until either the maximum number of epochs is reached or the improvement in the loss falls below `minChange`.

### Saving the Model

Now we'll save the winning estimator - available from the `base()` method of the search - so that we can use it in another process to predict the customers in our database. We wrap it in a [Persistent Model](https://rubixml.github.io/ML/3.0/persistent-model.html) meta-Estimator. The Persistent Model couples a [Persistable](https://rubixml.github.io/ML/3.0/persistable.html) estimator with a Persister so that we can save the trained model parameters to and load them from storage. In the example below we save the model to the filesystem using the default [RBX](https://rubixml.github.io/ML/3.0/serializers/rbx.html) serializer. RBX is a proprietary format that builds on PHP's native serialization by adding compression, integrity checking, and version compatibility detection. You could also use the standard PHP [Native](https://rubixml.github.io/ML/3.0/serializers/native.html) serializer if you wanted to.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel($estimator->base(), new Filesystem('model.rbx'));

$estimator->save();
```

Together, the training script produces three artifacts: `training.csv` and `testing.csv`, which hold the stratified train/test split we exported earlier, and `model.rbx`, which holds the trained Logit Boost model parameters. Keeping the split on disk means we can re-score the saved model at any time without having to train again - which is exactly what we'll do in the next section.

### Validating the Model

The operation of making predictions is referred to as "inference" in Machine Learning terms because it involves taking an unlabeled sample and inferring its label. We're going to need to generate some test predictions from the held-out testing set in order to validate the model. To keep the training script focused on training, we perform validation in a separate script called `validate.php`. Because the train/test split and the trained model were both persisted to disk, `validate.php` can be re-run at any time to score the saved model without having to train again.

We'll start by rebuilding the exact testing set we exported during training by passing `testing.csv` to the CSV extractor. The file contains the features followed by the label as the last column, so we can instantiate a Labeled dataset from it directly.

```php
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;

$extractor = new CSV('testing.csv');

$dataset = Labeled::fromIterator($extractor);
```

Next, we load the trained model from storage by calling the `load()` method on the Persistent Model meta-class with a Filesystem persister pointing to the path of its file in storage. Note that you may have to supply an optional Serializer if the default one wasn't used. Once loaded from storage, the model is ready to go in the same state that it was saved in.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('model.rbx'));
```

Before making predictions, we apply the same Float Type Converter we used during training so that the samples are preprocessed in exactly the same way. Since the converter is stateless, a freshly constructed instance converts the numeric strings identically to the one used in training - there is no fitted transformer that needs to be loaded from storage. Finally, we return the predictions by passing the dataset to the `predict()` method on the estimator. The predictions come back in the same order as the samples we loaded from the file.

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());

$predictions = $estimator->predict($dataset);
```

Under the hood, the Logit Boost algorithm sums the predictions of every booster in the ensemble to compute a margin `z` for the unknown sample, then passes that margin through the [Sigmoid](https://rubixml.github.io/ML/3.0/neural-network/activation-functions/sigmoid.html) activation function to squash it into a probability between 0 and 1. The sample is assigned the class on the positive side of the decision boundary - that is, whichever class corresponds to a positive margin. Because inference is just an accumulation of the individual tree predictions followed by a single squashing function, the calculation remains numerically stable no matter how many epochs were trained.

With the test predictions and their ground-truth labels in hand, we can now turn our focus to validating the model using the "holdout" technique. The process we use to determine generalization performance is called cross-validation and the holdout technique is one of the most straightforward approaches. The upside to this method is that it's quick and only requires training one model to produce a meaningful validation score. However, the downside to this technique is that, since the validation score for the model is only calculated from a portion of the samples, it has less coverage than methods that train multiple models and test them on different samples each time. In the next example, we're going to generate a report from the held out testing data that contains detailed metrics for us to evaluate the accuracy of the model.

We'll instantiate a [Multiclass Breakdown](https://rubixml.github.io/ML/3.0/cross-validation/reports/multiclass-breakdown.html) and [Confusion Matrix](https://rubixml.github.io/ML/3.0/cross-validation/reports/confusion-matrix.html) report generator and wrap them in an [Aggregate Report](https://rubixml.github.io/ML/3.0/cross-validation/reports/aggregate-report.html) so they can be generated at the same time. Multiclass Breakdown is a detailed report containing scores for a multitude of metrics including Accuracy, Precision, Recall, F-1 Score, and more on an overall and per-class basis. Confusion Matrix is a table that pairs the predictions counts on one axis with their ground-truth counts on the other. Counting each pair gives us a sense for which classes the estimator might be "confusing" another class for.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$reportGenerator = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);
```

To create the report object call the `generate()` method on the report generator with the predictions we generated from the testing set and the ground-truth labels from the dataset as arguments.

```php
$report = $reportGenerator->generate($predictions, $dataset->labels());
```

Since the Report object implements the [Stringable](https://www.php.net/manual/en/class.stringable.php) interface, we can output the report by echoing it out directly to the terminal. The example below illustrates a typical report for this classifier and dataset. You'll notice that Logit Boost did a pretty good job at distinguishing the churned customers with an accuracy of about 81%.

```php
echo $report;
```

```json
[
    {
        "overall": {
            "accuracy": 0.8113475177304964,
            "balanced accuracy": 0.6785559432618256,
            "f1 score": 0.7044435128243115,
            "precision": 0.8011456628477905,
            "recall": 0.6785559432618256,
            "specificity": 0.6785559432618256,
            "negative predictive value": 0.8011456628477905,
            "false discovery rate": 0.19885433715220946,
            "miss rate": 0.3214440567381744,
            "fall out": 0.3214440567381744,
            "false omission rate": 0.19885433715220946,
            "mcc": 0.46377299571663266,
            "informedness": 0.3571118865236511,
            "markedness": 0.6022913256955811,
            "true positives": 572,
            "true negatives": 572,
            "false positives": 133,
            "false negatives": 133,
            "cardinality": 705
        },
        "classes": {
            "No": {
                "accuracy": 0.8113475177304964,
                "balanced accuracy": 0.6785559432618256,
                "f1 score": 0.8821966341895483,
                "precision": 0.8150572831423896,
                "recall": 0.9613899613899614,
                "specificity": 0.39572192513368987,
                "negative predictive value": 0.7872340425531915,
                "false discovery rate": 0.18494271685761043,
                "miss rate": 0.038610038610038644,
                "fall out": 0.6042780748663101,
                "false omission rate": 0.21276595744680848,
                "informedness": 0.3571118865236511,
                "markedness": 0.6022913256955811,
                "mcc": 0.46377299571663266,
                "true positives": 498,
                "true negatives": 74,
                "false positives": 113,
                "false negatives": 20,
                "cardinality": 518,
                "proportion": 0.7347517730496453
            },
            "Yes": {
                "accuracy": 0.8113475177304964,
                "balanced accuracy": 0.6785559432618256,
                "f1 score": 0.5266903914590747,
                "precision": 0.7872340425531915,
                "recall": 0.39572192513368987,
                "specificity": 0.9613899613899614,
                "negative predictive value": 0.8150572831423896,
                "false discovery rate": 0.21276595744680848,
                "miss rate": 0.6042780748663101,
                "fall out": 0.038610038610038644,
                "false omission rate": 0.18494271685761043,
                "informedness": 0.3571118865236511,
                "markedness": 0.6022913256955811,
                "mcc": 0.46377299571663266,
                "true positives": 74,
                "true negatives": 498,
                "false positives": 20,
                "false negatives": 113,
                "cardinality": 187,
                "proportion": 0.2652482269503546
            }
        }
    },
    {
        "No": {
            "No": 498,
            "Yes": 113
        },
        "Yes": {
            "No": 20,
            "Yes": 74
        }
    }
]
```

We can also save the report to share with our colleagues or look at later. To save the report, call the `saveTo()` method on the Encoding object that is returned by calling the `toJSON()` method on the Report object. In this example, we'll use the Filesystem Persister to save the report to a file named `report.json`.

```php
use Rubix\ML\Persisters\Filesystem;

$report->toJSON()->saveTo(new Filesystem('report.json'));
```

That's the whole of `validate.php` - run it with `php validate.php` to print the report to the terminal and write `report.json`, giving us a detailed snapshot of how well the saved model generalizes to customers it has never seen before.

### Going Into Production

In practice, we'd probably spend some more time iterating over training and cross-validation in an effort to fine-tune the dataset and hyper-parameters. For the next part of this tutorial, we'll assume that we're fine with the model performance so far and we're ready to put it into production.

First, we need to make the choice between doing real-time inference or caching the predictions. For this problem, it makes a lot of sense to generate predictions for all our customers at the same time and then storing the prediction in the database alongside the customer's data. Then, we could periodically predict the new customers and update the existing customers using a script that runs in the background of our application. The nice thing about this design is that we don't need to keep the model loaded into memory. However, if you need the prediction for new customers instantly or if you have a quickly evolving model, you may want to consider doing inference in real time. See the [Server](https://github.com/RubixML/Server) package for an example of how to do this in a performant way using asynchronous PHP and a long-running process.

We're going to start a new script for predicting the label of the customers in our database. For demonstration, we've provided an example Sqlite database with over 2000 customers. Let's load the samples from the database and use our saved model to predict the at-risk customers. The [SQL Table](https://rubixml.github.io/ML/3.0/extractors/sql-table.html) extractor is an iterator that iterates over an entire database table. In the next example, we'll pass a PDO object referencing our Sqlite database to the SQL Table extractor's constructor along with the name of the table we want to iterate over.

```php
use Rubix\ML\Extractors\SQLTable;
use PDO;

$connection = new PDO('sqlite:database.sqlite');

$extractor = new SQLTable($connection, 'customers');
```

If we didn't want to load all the customers in our database, we could wrap the extractor within the standard PHP Limit Iterator to specify an offset and a limit.

```php
$extractor = new LimitIterator($extractor->getIterator(), 0, 100);
```

As we did with the training and validation sets, we'll instantiate a Column Picker to select the features from the database to input to the estimator. We'll also include the `Id` column which we'll use later when we update the database with the predictions.

```php
$extractor = new ColumnPicker($extractor, [
    'Id', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MonthsInService', 'Phone',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'TV', 'Movies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Region',
]);
```

Now, instantiate an Unlabeled dataset object by calling the `fromIterator()` method with the extractor as an argument.

```php
$dataset = Unlabeled::fromIterator($extractor);
```

To return the customer ids for every sample in the dataset, call the `feature()` method with the column offset of 0. Avoid feeding the customer Id to the estimator by dropping the it from the dataset.

```php
$ids = $dataset->feature(0);

$dataset->dropFeature(0);
```

We're almost there! Now, let's load the model we saved earlier into memory by calling the `load()` method on the Persistent Model meta-class with a Filesystem persister pointing to the path of its file in storage. Note that you may have to supply an optional Serializer if the default one wasn't used. Once loaded from storage, the model is ready to go in the same state that it was saved in.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('model.rbx'));
```

Before making predictions, we apply the same Float Type Converter we used during training so the samples are preprocessed in exactly the same way. Since the converter is stateless, no fitted transformer needs to be loaded from storage - a freshly constructed instance converts the numeric strings identically.

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());
```

Finally, return the predictions for the customers in the database by passing the inference set to the `predict()` method on the estimator. The predictions will be returned in the same order as the samples we loaded from the database.

```php
$predictions = $estimator->predict($dataset);
```

We'll use the same PDO connection object from before to prepare a SQL statement to update the customer. From here, we can loop through the predictions and update the corresponding rows in the database.

```php
$statement = $connection->prepare("UPDATE customers SET churn=? WHERE id=?");

foreach ($predictions as $i => $prediction) {
    $statement->execute([$prediction, $ids[$i]]);
}
```

Voila! You've identified the customers that may be at risk of churning. Let's take a moment to recap. Remember we loaded a training dataset that had been labeled by our customer service department as either churning or not churning. Then we used that dataset to train a Logit Boost classifier to predict the churn rate of the customers in our database. Lastly, we stored those predictions in the database so we could use them later within our app. Nice work! For further learning you may want to consider ...

- Training with a different subset of the features. Are some features more predictive than others?
- How does the learning rate and the maximum number of epochs effect the predictions?
- Widening the grid search - would a larger search space, or a different validator or scoring metric, find even better hyper-parameters?
- Swapping Logit Boost for another classifier such as [Random Forest](https://rubixml.github.io/ML/3.0/classifiers/random-forest.html) or [Naive Bayes](https://rubixml.github.io/ML/3.0/classifiers/naive-bayes.html).

## Original Dataset

https://github.com/codebrain001/customer-churn-prediction

## License

The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
