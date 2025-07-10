# Vibrational Density of States for a system
## Sample command line argument for computing vdos for Li Al P S system

### CL Argument: 
```sh
CPDyAna vdos --data-dir "./data" --elements Li Al P S
```
```sh
CPDyAna vdos --data-dir "./data" --lammps-elements Ti La Li O --element-mapping 1:Li 2:La 3:Ti 4:O --lammps-timestep 1 --elements Li La Ti O  
```