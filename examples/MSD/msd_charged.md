# Mean Squared Displacement vs Time (Charged)
## Sample command line argument for msd vs time for Charged species.

### CL Argument

```sh
CPDyAna msd --data-dir . -T 600 --diffusing-elements Li --diffusivity-choices Charged --initial-slope-time 5 --final-slope-time 15 --block 500 --initial-time 0 --final-time 20 --first-time 0 --last-time 20 --diffusivity-direction-choices XYZ XY YZ ZX X Y Z
```

### Sample Output

```terminal
Starting MSD analysis...                                                                                                                                                       
Processing 1 file sets for 1 temperature(s)
Analyzing elements: Li

Saving results to OUTPUT.json...
Generating MSD plot with 7 data series...
MSD plot saved to: MSD.jpg
```