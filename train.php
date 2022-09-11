<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\IntervalDiscretizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$extractor = new ColumnPicker(new CSV('dataset.csv', true), [
    "Gender", "SeniorCitizen", "Partner", "Dependents", "MonthsInService", "Phone",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "TV", "Movies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]);

$dataset = Labeled::fromIterator($extractor)
    ->apply(new NumericStringConverter())
    ->apply(new IntervalDiscretizer(3));

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$estimator = new NaiveBayes([
    "Yes" => 0.05,
    "No" => 0.95,
]);

$estimator->train($training);

$probabilities = $estimator->proba($testing);

$metric = new ProbabilisticAccuracy();

$score = $metric->score($probabilities, $testing->labels());

$logger->info("Model is $score accurate");

$logger->info('Generating report');

$predictions = $estimator->predict($testing);

$reportGenerator = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$report = $reportGenerator->generate($predictions, $testing->labels());

$report->toJSON()->saveTo(new Filesystem('report.json'));

$logger->info('Report saved to report.json');

$estimator = new PersistentModel($estimator, new Filesystem('model.rbx'));

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();

    $logger->info('Model saved to model.rbx');
}
