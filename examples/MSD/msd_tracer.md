# Mean Squared Displacement vs Time (Tracer)
## Sample command line argument for msd vs time tracer species

### CL Argument 

```sh
CPDyAna msd --data-dir "./data" -T 600 -e Li --diffusivity-direction-choices XYZ XY YZ ZX X Y Z --diffusivity-choices Tracer --initial-time 0 --final-time 300 --initial-slope-time 5 --final-slope-time 200 --block 500 --first-time 0 --last-time 300 
```
```sh
CPDyAna msd --data-dir "./data" --lammps-elements Li La Ti O --element-mapping 1:Li 2:La 3:Ti 4:O --lammps-timestep 1 --initial-time 0 --final-time 2000 --initial-slope-time 300 --final-slope-time 1500 -T 600 --first-time 0 --last-time 2000 --diffusivity-choices Tracer --diffusivity-direction-choices XYZ XY YZ ZX X Y Z
```