# Velocity Autocorrelation Function
## Sample command line arugument for computing vaf.

### CL Argument: 
```sh
CPDyAna vaf --data-dir "./data" --element Li Al P S --t-start-fit-ps 5 --stepsize-t 1 --stepsize-tau 10 --t-end-fit-ps 100
```
```sh
CPDyAna vaf --data-dir "./data" --lammps-elements Li La Ti O --element-mapping 1:Li 2:La 3:Ti 4:O --lammps-timestep 1 --element Li --t-end-fit-ps 1000
```