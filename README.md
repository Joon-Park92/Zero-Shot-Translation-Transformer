# A TensorFlow Zero-Shot translation implemented by Transformer 
!image[https://1.bp.blogspot.com/-jwgtcgkgG2o/WDSBrwu9jeI/AAAAAAAABbM/2Eobq-N9_nYeAdeH-sB_NZGbhyoSWgReACLcB/s640/image01.gif]

## Reference 
  * https://github.com/Kyubyong/transformer
  * Jörg Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
  * P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
  
## Requirements
  
## File description
  * `preprocessor.py` download & preprocess data ( you can use OpenSubtitle2018 / MultiUN dataset )
  * `model.py` define transformer model
  * `data_loader.py` define class for dataload
  * `model_layer.py` has all layers need for transformer
  * `train.py` for training model
  * `eval.py` is for evaluation.

## Training
* STEP 1. Edit the hyperparams.py
* STEP 2. Run preprocessor.py , it will preprocess and make training data 
* STEP 3. Run train.py to train your model
* STEP 4. Run `eval.py` to save result
