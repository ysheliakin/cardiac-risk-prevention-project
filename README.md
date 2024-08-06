random_undersampling.py is used with the initial unbalaned dataset available on kaggle. This is what normalizes the data for model training.
progress_update2.py is the notebook used for testing 6 model types. This is launched using the CAP4770 docker instance
with Jupyter notebook.
7_models_with_tuning.py is a script with detailed comments where the subsequent training occurred. It outputs the results in json format and the models with .joblib extension for testing with the next script:
reintroduce_unbalanced_dataset.py tests the models produced by the previous script against the original dataset
compare_model_results.py loads in the json results from the balanced and unbalanced datasets and produces figures to compare the results.
visualize_tuning_effects.py utilizes the json output during the model training to generate figures that show the impact parameter tuning had on prediction accuracy.