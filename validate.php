<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

$logger = new Screen();

$extractor = new CSV('testing.csv');

$estimator = PersistentModel::load(new Filesystem('model.rbx'));

$reportGenerator = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$logger->info('Making predictions');

$dataset = Labeled::fromIterator($extractor);

$dataset->apply(new FloatTypeConverter());

$predictions = $estimator->predict($dataset);

$report = $reportGenerator->generate($predictions, $dataset->labels());

echo $report;

$report->toJSON()->saveTo(new Filesystem('report.json'));

$logger->info('Report saved as report.json');
