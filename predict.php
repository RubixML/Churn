<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Extractors\SqlTable;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$connection = new PDO('sqlite:database.sqlite');

$extractor = new SqlTable($connection, 'customers');

$extractor = new ColumnPicker($extractor, [
    'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MonthsInService', 'Phone',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'TV', 'Movies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Region',
]);

$dataset = Unlabeled::fromIterator($extractor);

$logger->info('Loading model into memory');

$estimator = PersistentModel::load(new Filesystem('model.rbx'));

$logger->info('Making predictions');

$predictions = $estimator->predict($dataset);

print_r($predictions);
