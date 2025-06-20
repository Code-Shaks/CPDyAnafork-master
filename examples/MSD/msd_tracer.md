# Mean Squared Displacement vs Time (Tracer)
## Sample command line argument for msd vs time tracer species

### CL Argument 

```sh
CPDyAna msd --data-dir . -T 600 -e Li --diffusivity-direction-choices XYZ XY YZ ZX X Y Z --diffusivity-choices Tracer --initial-time 0 --final-time 300 --initial-slope-time 5 --final-slope-time 200 --block 500 --first-time 0 --last-time 300 
```

### Sample output
```terminal
Starting MSD analysis...                                                                                                                                                       
Processing 1 file sets for 1 temperature(s)                                     
Analyzing elements: Li

Saving results to OUTPUT.json...
Generating MSD plot with 7 data series...
MSD plot saved to: MSD.jpg
```