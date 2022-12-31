<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\IntervalDiscretizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$extractor = new ColumnPicker(new CSV('dataset.csv', true), [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MonthsInService', 'Phone',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'TV', 'Movies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Region', 'Churn',
]);

$dataset = Labeled::fromIterator($extractor);

[$training, $testing] = $dataset->randomize()->stratifiedSplit(0.8);

$estimator = new NaiveBayes([
    'Yes' => 0.1,
    'No' => 0.9,
]);

$estimator = new Pipeline([
    new NumericStringConverter(),
    new IntervalDiscretizer(3, true),
], $estimator);

$logger->info('Training the model');

$estimator->train($training);

$logger->info('Making predictions');

$predictions = $estimator->predict($testing);

$reportGenerator = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$report = $reportGenerator->generate($predictions, $testing->labels());

echo $report;

$report->toJSON()->saveTo(new Filesystem('report.json'));

$logger->info('Report saved as report.json');

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator = new PersistentModel($estimator, new Filesystem('model.rbx'));

    $estimator->save();

    $logger->info('Model saved as model.rbx');
}
