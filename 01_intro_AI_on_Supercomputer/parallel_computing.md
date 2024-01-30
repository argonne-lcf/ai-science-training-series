# Concept of Parallel Computing  - pi examples
![PI](https://www.101computing.net/wp/wp-content/uploads/estimating-pi-monte-carlo-method.png)

```python
from mpi4py import MPI
import numpy as np
import random
import time
comm = MPI.COMM_WORLD

N = 5000000
Nin = 0
t0 = time.time()
for i in range(comm.rank, N, comm.size):
    x = random.uniform(-0.5, 0.5)
    y = random.uniform(-0.5, 0.5)
    if (np.sqrt(x*x + y*y) < 0.5):
        Nin += 1
res = np.array(Nin, dtype='d')
res_tot = np.array(Nin, dtype='d')
comm.Allreduce(res, res_tot, op=MPI.SUM)
t1 = time.time()
if comm.rank==0:
    print(res_tot/float(N/4.0))
    print("Time: %s" %(t1 - t0))
```

```bash
ssh <username>@polaris.alcf.anl.gov
qsub -A ALCFAITP -l select=1 -q ALCFAITP -l walltime=0:10:00 -l filesystems=home:eagle
module load conda/2023-10-04
conda activate /soft/datascience/ALCFAITP/2023-10-04
cd YOUR_GITHUP_REPO
mpirun -np 1 python pi.py   # 3.141988,   8.029037714004517  s
mpirun -np 2 python pi.py   # 3.1415096   4.212774038314819  s
mpirun -np 4 python pi.py   # 3.1425632   2.093632459640503  s
mpirun -np 8 python pi.py   # 3.1411632   1.0610620975494385 s
```
