# BERT on Graphcore 

These instructions are to train a BERT Pytorch model on the POD16. 

Go to directory with BERT example 
```bash
cd ~/graphcore/examples/nlp/bert/pytorch
```


Activate PopTorch Environment 

```bash
source ~/venvs/graphcore/poptorch33_env/bin/activate
```

Install Requirements 

```bash
pip3 install -r requirements.txt
```

Run BERT on 4 IPUs (single Instance)
```bash
/opt/slurm/bin/srun --ipus=4 python3 pretraining_data.py --ipus-per-replica 4 --replication-factor 1 --training-steps 10 --dataset 'generated' 
```

<details>
  <summary>Sample Output</summary>
  
  ```bash
>  /opt/slurm/bin/srun --ipus=4 python3 pretraining_data.py --ipus-per-replica 4 --replication-factor 1 --training-steps 10 --dataset 'generated' 
srun: job 20100 queued and waiting for resources
srun: job 20100 has been allocated resources

You are executing bert_data directly.
Let's read the first input from sample dataset.
dataset length:  45 

input_ids (128,) int64 <class 'numpy.ndarray'> [  101  4438  9866  1010  1044 22571 10450  2050  2841  1012  2004  1996
  3418   103  1011  1011  1996  2171  2003  4895  5714  6442  4630  2000
  1037   103  1011  1011   103  2300 22390   103  2002  2453  2817  2011
   103  1010  2061   103  1010  1996  6697  1997 11965  2015  1998 11498
 19454  2015  1010  2012  1996 29203  4303  1997  8682  8835  1011  2282
  1010 10047   103  2063   102  1037 12125   103 16968  2993  1012  1037
  5357  2732  1010  1045  2572  1010  2909   999  1005  2506   103  1010
   103 14547  1010 14943 20946  8155   103  1011  1011  1005  1037  5357
  2732   999  1011  1011  5357  1010   103  1996   103  1997  4695  2097
  3499  1997  1996 28684  2571  1010  2426   103 27589  2015 24078  1996
   103  2088  1011  1011  5262  1010  2130   102] 


input_mask (128,) int64 <class 'numpy.ndarray'> [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] 


segment_ids (128,) int64 <class 'numpy.ndarray'> [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] 


masked_lm_positions (20,) int64 <class 'numpy.ndarray'> [ 13  25  28  31  36  39  53  56  62  67  82  84  88  90  98 102 104 115
 118 120] 


masked_lm_ids (20,) int64 <class 'numpy.ndarray'> [10878  8284 16486  2008  2154  1045  6730  2014 28065  1997  2002 25636
  2010  4308  1011  2065 13372  1996  1997  2896] 


next_sentence_labels (1,) int64 <class 'numpy.ndarray'> [0] 


And now, we are going to decode the tokens.

Downloading: 100%|██████████| 48.0/48.0 [00:00<00:00, 224kB/s]
Downloading: 100%|██████████| 226k/226k [00:00<00:00, 5.92MB/s]
Downloading: 100%|██████████| 455k/455k [00:00<00:00, 6.46MB/s]
Downloading: 100%|██████████| 570/570 [00:00<00:00, 6.79MB/s]


 [CLS] classic wisdom, hypatia herself. as the ancient [MASK] - - the name is unimportant to a [MASK] - - [MASK] water nightly [MASK] he might study by [MASK], so [MASK], the guardian of cloaks and parasols, at the fantasia doors of blew lecture - room, im [MASK]e [SEP] a spark [MASK] divinity itself. a fallen star, i am, sir!'continued [MASK], [MASK]ively, stroking decker lean [MASK] - -'a fallen star! - - fallen, [MASK] the [MASK] of philosophy will allow of the simile, among [MASK] hogs basics the [MASK] world - - indeed, even [SEP] 

  ```
</details>

