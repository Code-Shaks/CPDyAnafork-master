# van Hove correlation Function
## Sample command line argument for computing van hove correlation function for Li, both self and distinct. 

### CL Argument: 
```sh
CPDyAna vh --data-dir "./data" -T 600 -e Li --correlation Self Distinct --initial-time 0 --final-time 100                                                                             
```
```sh
CPDyAna vh --data-dir "./data" --lammps-elements Li La Ti O --element-mapping 1:Li 2:La 3:Ti 4:O --lammps-timestep 1 --export-verification -T 800 --initial-time 1 --final-time 10 --step-skip 1 --ngrid 10001 --sigma 0.01
```