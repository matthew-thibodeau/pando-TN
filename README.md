# Automated quantum tensor network design
Automated design of tensor networks for simulating condensed matter models, chemistry, and quantum circuits.

Dependencies:
* Scipy
* Quimb
* DGL

Contact:
* matthew.thibodeau@intel.com
* nicolas.sawaya@intel.com

Contributors:
Matthew Thibodeau, Subrata Goswami, Nicolas Sawaya.


## Numerical study

We optimize the topology and bond dimension in Tree Tensor Networks (TNN)  using the
Simulated Annealing (SA) algorithm.  
To launch a single optimization run, from folder `<repo>/tnn_learning/` type:
```bash
python3 ttn_perform_SA.py -L 16 -m 3 -d 1.0 -r 5 -i 100 -s 1234
```
where the options are:
- `L`: size of the Hamiltonian
- `m`: max bond dimension
- `d`: disorder strength
- `r`: number of runs (i.e. of different random values of the field)
- `i`: number of iterations of the optimizer (here SA)
- `s`: seed for the random number generator

The same result is obtained from the mnain repo folder by typing:
```bash
make sa ARGS="-L 16 -m 3 -d 1.0 -r 5 -i 100 -s 1234"
```
