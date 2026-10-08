<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\GridSearch;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\LogitBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

$logger = new Screen();

$extractor = new ColumnPicker(new CSV('dataset.csv', true), [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MonthsInService', 'Phone',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'TV', 'Movies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Region', 'Churn',
]);

$estimator = GridSearch::fromNamedParams(
    class: LogitBoost::class,
    params: [
        'booster' => [new RegressionTree(3), new RegressionTree(4)],
        'rate' => [0.1, 0.3],
        'ratio' => [0.3, 0.5, 0.7],
    ],
);

$estimator->setBackend(new Amp());

$estimator->setLogger($logger);

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator($extractor);

$logger->info('Preprocessing the dataset');

$dataset->apply(new FloatTypeConverter());

[$training, $testing] = $dataset->stratifiedSplit(0.9);

$estimator->setup(function (LogitBoost $estimator) use ($testing) {
    $estimator->setValidationDataset($testing);
});

$logger->info('Exporting train/test split');

$training->exportTo(new CSV('training.csv'), overwrite: true);
$testing->exportTo(new CSV('testing.csv'), overwrite: true);

$logger->info('Training the model');

$estimator->train($training);

echo $estimator->results();

$estimator->results()->toJSON()->saveTo(new Filesystem('results.json'));

$logger->info('Results saved to results.json');

$estimator = new PersistentModel($estimator->base(), new Filesystem('model.rbx'));

$estimator->save();

$logger->info('Model saved as model.rbx');
