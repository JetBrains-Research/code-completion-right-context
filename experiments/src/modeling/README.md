# Modeling

Python package with machine learning model implementation, training and logging routines.

## How to add new model?

Each model has to be subclass of BaseModel from base_model package.
Main method for training is forward.
Main method for inference is get_next_token_scores.
You have to implement cache support (`use_cache` = True and `past` = previous model step)
to speed up generation process.

If you want to train model with `catalyst` it is recommended to 
implement subclass of BaseConfigInitializer.
Usually, you need to implement only `init_model` and `init_logging_callback` methods.
Result of `init_all` method is dict with all entities for catalyst `SupervisedRunner`.

## Datasets

Some datasets are imlemented in `dataset.py` module:
* LanguageModelDataset - standard language model dataset, takes as input list of tokenized documents
or one big joined tokenized document.
* LanguageModelChunkDataset - dataset for situations, when you can't store all document in RAM,
each sequence has to be stored in its own file.

