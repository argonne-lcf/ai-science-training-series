# BERT on Graphcore 

These instructions are to train a BERT Pytorch model on the POD16. 

* Go to directory with BERT example 
  ```bash
  cd ~/graphcore/examples/nlp/bert/pytorch
  ```


* Activate PopTorch Environment 

  ```bash
  source ~/venvs/graphcore/poptorch33_env/bin/activate
  ```

* Install Requirements 

  ```bash
  pip3 install -r requirements.txt
  ```

* Run BERT on 4 IPUs (single Instance)
  ```bash
  /opt/slurm/bin/srun --ipus=4 python3 pretraining_data.py \
  --ipus-per-replica 4 --replication-factor 1 \
  --training-steps 10 --dataset 'generated' 
  ```

  <details>
    <summary>Sample Output</summary>
    
    ```bash
    
    >   /opt/slurm/bin/srun --ipus=4 python3 pretraining_data.py   --ipus-per-replica 4 --replication-factor 1   --training-steps 10 --dataset 'generated' 
    srun: job 20248 queued and waiting for resources
    srun: job 20248 has been allocated resources

    You are executing bert_data directly.
    Let's read the first input from sample dataset.
    dataset length:  45 

    input_ids (128,) int64 <class 'numpy.ndarray'> [  101  4298  2023  2089  2031  2042   103  3114  2339  2220  4125  2869
      1999   103 10246  1010   103  1996 16373   103  1010  4233  1037 16465
    10427  1997  2303   103  1998 15839   103  2037  2159  2000  1996 16931
      2098  2030  2634  1011 10710  8871 15717  2682  2068  1012   102  2025
      2007  1037  3193  2000  5456  1012  1037 17271  1999  2010  6644  4412
      1010  1011  1011  3243   103  2007  2010   103  1010   103 12298  5178
      3372 14243  1010  1011  1011  2018   103  2094  2032  2012  1018  1037
      1012  1049   103  1010  2007  1037 10361  1000 25277  1000   103  4954
    15019  1012   103 11772  2013  2010   103  8632  4188  2000   103  2571
      1037  2543   103  4318  2010  2793  1011  4253  1010  1998  2002  2018
    28667 22957  2063  2000  1037  2062  3073   102] 


    input_mask (128,) int64 <class 'numpy.ndarray'> [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] 


    segment_ids (128,) int64 <class 'numpy.ndarray'> [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] 


    masked_lm_positions (20,) int64 <class 'numpy.ndarray'> [  2   6  13  16  19  27  30  56  64  67  69  72  78  86  94  98 102 106
    109 110] 


    masked_lm_ids (20,) int64 <class 'numpy.ndarray'> [ 2023  1996  2008  2076  2161  1010  4196  1999  8335 23358 17727  3372
    27384  1012  1998  1996  3536  2785  2543  2000] 


    next_sentence_labels (1,) int64 <class 'numpy.ndarray'> [0] 


    And now, we are going to decode the tokens.



    [CLS] possibly this may have been [MASK] reason why early risers in [MASK] locality, [MASK] the rainy [MASK], adopted a thoughtful habit of body [MASK] and seldom [MASK] their eyes to the rifted or india - ink washed skies above them. [SEP] not with a view to discovery. a leak in his cabin roof, - - quite [MASK] with his [MASK], [MASK]rovident habits, - - had [MASK]d him at 4 a. m [MASK], with a flooded " bunk " [MASK] wet blankets. [MASK] chips from his [MASK] pile refused to [MASK]le a fire [MASK] dry his bed - clothes, and he had recourse to a more provide [SEP] 

    ```
  </details>

