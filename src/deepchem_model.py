import argparse

import numpy as np

from sklearn.metrics import mean_squared_error

import deepchem as dc

import matplotlib
import joblib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("Evaluate Machine Learning Approaches for Modeling Elution Times")
parser.add_argument("model_name")
parser.add_argument("--featurizer", type=str, default='weave',
                    help='weave, conv, or ecfp')
parser.add_argument("--split", type=str, default='random', 
                    help="random, index, or scaffold")
parser.add_argument("--model", type=str, default='mpnn', 
                    help="mpnn, or gconv")
parser.add_argument("--batch-size", type=int, default=64,
                    help="Batch size to use during training")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate to use during training")
parser.add_argument("-T", type=int, default=3,
                    help="Number of message passing steps in MPNN")
parser.add_argument("-M", type=int, default=5,
                    help="Number of computation steps in the set2set readout \
                          readout model in the MPNN")

args = parser.parse_args()

# Featurization
args.featurizer = args.featurizer.lower()
featurizer_options = {
        'weave': dc.feat.WeaveFeaturizer(),
        'conv': dc.feat.ConvMolFeaturizer(),
        'ecfp': dc.feat.CircularFingerprint(size=1024),        
}
featurizer = featurizer_options[args.featurizer]
print("Using featurizer: {:s}".format(args.featurizer))

loader = dc.data.CSVLoader(tasks=["neg RT (min)"], smiles_field="smiles", 
                                  featurizer=featurizer,
                                  verbose=True,
                                  log_every_n=10)
raw_dataset = loader.featurize('../data/combined_time.csv')
raw_dataset = dc.data.datasets.NumpyDataset.from_DiskDataset(raw_dataset)

# Splitting
args.split = args.split.lower()
splitters = {
        'index': dc.splits.IndexSplitter(),
        'random': dc.splits.RandomSplitter(),
        'scaffold': dc.splits.ScaffoldSplitter()
}
print("Using splitter: {:s}".format(args.split))
splitter = splitters[args.split]
raw_training_data, raw_testing_data = splitter.train_test_split(raw_dataset)

# Transformation
transformers = [
    dc.trans.NormalizationTransformer(transform_y=True, dataset=raw_training_data)]


transformed_data = []
for dataset in [raw_training_data, raw_testing_data]:
    for transformer in transformers:
          transformed_data.append(transformer.transform(dataset))
training_data, testing_data = tuple(transformed_data)

# Model Building
args.model = args.model.lower()
if args.model == 'mpnn':
    n_atom_feat = 75
    n_pair_feat = 14
    model = dc.models.MPNNTensorGraph(
        n_tasks=1,
        mode="regression",
        batch_size=args.batch_size,
        model_dir=args.model_name,
        verbosity="high",
        
        learning_rate=args.lr,
        use_queue=False,
        n_atom_feat=n_atom_feat,
        n_pair_feat=n_pair_feat,
        T=args.T,
        M=args.M,

    )
elif args.model == 'gconv':
    model = dc.models.GraphConvTensorGraph(
        n_tasks=1,
        mode='regression',
        batch_size=args.batch_size,
        model_dir=args.model_name,
        verbosity='high',
    )
elif args.model == 'ann':
    model = dc.models.MultitaskRegressor(
        n_tasks=1,
        mode='regression',
        batch_size=args.batch_size,
        model_dir=args.model_name,
        verbosity='high',
        
        n_features=training_data.X.shape[1],
        layer_sizes=[300],
        dropouts=[.25],
        learning_rate=0.001,
    )
else:
    raise ValueError("Unknown model type")
print("Using model: {:s}".format(args.model))

# Training
#model.optimizer = dc.models.tensorgraph.optimizers.Adam(learning_rate=args.lr,
#                                                        beta1=0.9,
#                                                        beta2=0.999,
#                                                        epsilon=1e-08)
model.fit(training_data, nb_epoch=100, verbose=True)
model.save()

# Evaluation
metric = dc.metrics.Metric(dc.metrics.rms_score, mode='regression')
metric = dc.metrics.Metric(dc.metrics.rms_score, mode='regression')
print("Evaluating model")
train_scores = model.evaluate(training_data, [metric])
test_scores = model.evaluate(testing_data, [metric])

training_predictions = model.predict(training_data)
testing_predictions = model.predict(testing_data)

untransformed_data = []
for dataset in [training_predictions, testing_predictions]:
    for transformer in transformers[-1::-1]:
          untransformed_data.append(transformer.untransform(dataset))
raw_training_predictions, raw_testing_predictions = tuple(untransformed_data)

y_expt = raw_training_data.y
y_expt = y_expt.flatten()
y_pred = raw_training_predictions
y_pred = y_pred.flatten()
fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.add_subplot(111)
ax1.plot(y_expt, y_pred, 'b.')
ax1.plot(np.arange(0, max(y_expt)), np.arange(0, max(y_expt)), 'r-')
ax1.set_xlabel("Experimental Elution Time [min]")
ax1.set_ylabel("Predicted Elution Time [min]")
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax1.set_title("Training Dataset (n={:d}, RMSE={:.2f} min)".format(
    len(y_expt), 
    np.sqrt(mean_squared_error(y_expt, y_pred))))
fig1.savefig(model.model_dir + "/Training_Performance.png")

bad_molecules = raw_training_data.ids[(np.abs(y_pred - y_expt)) > 2.0]
print("On Training Data:")
print("Training Dataset (n={:d}, RMSE={:.2f} min)".format(
    len(y_expt), 
    np.sqrt(mean_squared_error(y_expt, y_pred))))
print("Training RMS Score: %f" % train_scores["rms_score"])
print(f"Percentage of molecules with error > 2 minutes: {len(bad_molecules)*100/len(y_expt):.1f}%")
print(f"Number of molecules with error > 2 minutes: {len(bad_molecules):d}")

y_expt = raw_testing_data.y
y_expt = y_expt.flatten()
y_pred = raw_testing_predictions
y_pred = y_pred.flatten()
fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.add_subplot(111)
ax1.plot(y_expt, y_pred, 'b.')
ax1.plot(np.arange(0, max(y_expt)), np.arange(0, max(y_expt)), 'r-')
ax1.set_xlabel("Experimental Elution Time [min]")
ax1.set_ylabel("Predicted Elution Time [min]")
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax1.set_title("Test Dataset (n={:d}, RMSE={:.2f} min)".format(
    len(y_expt), 
    np.sqrt(mean_squared_error(y_expt, y_pred))))
fig1.savefig(model.model_dir + "/Test_Performance.png")

bad_molecules = raw_testing_data.ids[(np.abs(y_pred - y_expt)) > 2.0]
print("On Testing Data:")
print("Test Dataset (n={:d}, RMSE={:.2f} min)".format(
    len(y_expt), 
    np.sqrt(mean_squared_error(y_expt, y_pred))))
print("Testing RMS Score: %f" % test_scores["rms_score"])
print(f"Percentage of molecules with error > 2 minutes: {len(bad_molecules)*100/len(y_expt):.1f}%")
print(f"Number of molecules with error > 2 minutes: {len(bad_molecules):d}")

joblib.dump(raw_training_data, model.model_dir+ "/training.joblib")
joblib.dump(raw_testing_data, model.model_dir + "/testing.joblib")
joblib.dump(transformers, model.model_dir + "/transformers.joblib")
model.save()

