# Make Cooperative Binding Model

## Installing

## Generating cooperative model

### Useful import


### Reading input training data

The `flip_one_face_orientation` makes sure all sequences with HT/TH orientations
face the same direction.
~~~~{.python}
trainingpath = "input/training_data/training_p01_adjusted.tsv"
df = pd.read_csv(trainingpath, sep="\t")
traindf = Training(df, corelen=4).flip_one_face_orientation(["GGAA","GGAT"])
~~~~

### Create an imads model
~~~~{.python}
imads_paths = ["input/imads_model/Ets1_w12_GGAA.model", "input/imads_model/Ets1_w12_GGAT.model"]
imads_cores = ["GGAA", "GGAT"]
imads_models = [iMADSModel(path, core, 12, [1, 2, 3])
                for path, core in
                zip(imads_paths, imads_cores)]
imads = iMADS(imads_models, 0.2128)
~~~~

### Define random forest parameter

~~~~{.python}
rf_param_dict = {
    'n_estimators': [1000, 500, 1000, 1500],
    'max_depth':[5, 10, 15],
    "min_samples_leaf" : [10, 15, 20],
    "min_samples_split" :[10, 15 ,20]
}
~~~~

### Define model to use

~~~~{.python}
best_models = {
    "dist-ori-affinity":
    BestModel(clf="RF",
                 param_dict=rf_param_dict,
                 train_data=t.get_training_df({
                         "distance":{"type":"numerical"},
                         "sitepref": {"imadsmodel": imads, "modelwidth":12},
                         "orientation": {"positive_cores":["GGAA","GGAT"], "one_hot":True}
                     })
     ).run_all()
}
~~~~

### Get the model performance

~~~~{.python}
plot_metrics(best_models, "Average ROC Curves Using RF", "auc.png", score_type=score_type)
~~~~
