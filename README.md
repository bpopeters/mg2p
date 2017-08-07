# mg2p
## Tools for multilingual grapheme-to-phoneme conversion

This is a set of high-level utilities for building data sets for multilingual g2p systems. It also includes support for training 
and preprocessing those models using [OpenNMT](http://opennmt.net), and for computing g2p error metrics on the test set. I used
it to create the models for my paper for the Workshop on Building Linguistically Generalizable NLP Systems at EMNLP 2017 (paper
preprint [here](https://arxiv.org/abs/1708.01464)).

The basic usage is like this:

```
python mg2p.py spanish-model -preprocess 
```

Where the first argument is the name of the model you want to create. The flags `-preprocess`, `-train`, `-test`, or any
combination of these three may follow, depending on how much of the process you want to do in a single command. Other optional
arguments can specify which languages and scripts to include in the training/validation/test data and what features to append
to the source and target.

Pronunciation data for the models is available [here](https://drive.google.com/drive/folders/0B7R_gATfZJ2aSlJabDMweU14TzA).

NOTE: Due to (over)zealous adoption of language identification feature embeddings in the months since I wrote the paper, mg2p.py
does not actually currently support the language identification token approach described in the paper. This will be rectified soon.
