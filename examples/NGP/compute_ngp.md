# Non-Gaussian Parameter
## Sample command line argument example for NGP

## CL Argument
```sh
CPDyAna ngp --data-dir "./data" --lammps-elements Li La Ti O --element-mapping 1:Li 2:La 3:Ti 4:O --lammps-timestep 1 --initial-time 0 --final-time 2000 --step-skip 1 -T 600 --first-time 0 --last-time 2000  
```
```sh
CPDyAna ngp --data-dir "./data" --element Li --initial-time 0 --final-time 200 --step-skip 1 -T 600 --first-time 0 --last-time 200 
```