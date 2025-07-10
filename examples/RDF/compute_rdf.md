# Radial Distribuition Function for a system
## Sample command line argument for computing RDF for Li Al P S

### CL Argument: 
```sh
CPDyAna rdf --data-dir . xlim 0 10
```
```sh
CPDyAna rdf --data-dir "./data" --lammps-elements Ti La Li O --element-mapping 1:Li 2:La 3:Ti 4:O --lammps-timestep 1 --central-atom Li La Ti O --pair-atoms Li La Ti O
```