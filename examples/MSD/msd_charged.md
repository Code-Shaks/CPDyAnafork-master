# Mean Squared Displacement vs Time (Charged)
## Sample command line argument for msd vs time for Charged species.

### CL Argument

```sh
CPDyAna msd --data-dir . -T 600 --diffusing-elements Li --diffusivity-choices Charged --initial-slope-time 5 --final-slope-time 15 --block 500 --initial-time 0 --final-time 20 --first-time 0 --last-time 20 --diffusivity-direction-choices XYZ XY YZ ZX X Y Z
```
```sh
CPDyAna msd --data-dir "./data" --lammps-elements Li La Ti O --element-mapping 1:Li 2:La 3:Ti 4:O --lammps-timestep 1 --initial-time 0 --final-time 2000 --initial-slope-time 300 --final-slope-time 1500 -T 600 --first-time 0 --last-time 2000 --diffusivity-choices Charged --diffusivity-direction-choices XYZ XY YZ ZX X Y Z
```