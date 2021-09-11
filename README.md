# lstm-language-model
LSTM-based language model for text generation using pytorch



## Training
```
    > python train.py
```

## Arguments
    
```
    '--max-epochs'        >>>>      NUMBER OF EPOCHS, type=int, default=20
    '--batch-size'        >>>>      BATCH SIZE FOR TRAINIG type=int, default=256
    '--sequence-length'   >>>>      LENGTH OF SEQUENCE PASSED TO LSTM, type=int, default=8
```

## Dataset

Jokes crawled from reddit. could be replaced by any corpus in the following format (CSV file name also has to be changed in `dataset.py`)
```  
    id,text
    1,...
    2,...
    
```
## Inference

Seed text for completion and other inference setting is in `train.py`.

generated Sentence is stored in `output.txt`
