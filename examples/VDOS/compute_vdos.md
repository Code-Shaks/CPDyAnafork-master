# Vibrational Density of States for a system
## Sample command line argument for computing vdos for Li Al P S system

### CL Argument: 
```sh
CPDyAna vdos --data-dir . --elements Li Al P S
```

### Sample output
```terminal
Starting VDOS analysis for elements: Li, Al, P, S
Processing 1 file sets...
Processing file set 1/1: LiAlPS
Built 220747 frames with velocities.
nblocks = 1, blocks.shape = (1, 220747, 2, 3), block_length_ps = 427169.72717
nblocks = 1, blocks.shape = (1, 220747, 22, 3), block_length_ps = 427169.72717
nblocks = 1, blocks.shape = (1, 220747, 4, 3), block_length_ps = 427169.72717
nblocks = 1, blocks.shape = (1, 220747, 24, 3), block_length_ps = 427169.72717
Saving VDOS plot to: D:\Internship\Summer Internship 2025\CPDyAnafork-master\CPDyAnafork-master\vdos_LiAlPS_1.png
nblocks = 1, blocks.shape = (1, 220747, 2, 3), block_length_ps = 427169.72717
nblocks = 1, blocks.shape = (1, 220747, 22, 3), block_length_ps = 427169.72717
nblocks = 1, blocks.shape = (1, 220747, 4, 3), block_length_ps = 427169.72717
nblocks = 1, blocks.shape = (1, 220747, 24, 3), block_length_ps = 427169.72717
Saving custom VDOS plot to: D:\Internship\Summer Internship 2025\CPDyAnafork-master\CPDyAnafork-master\vdos_LiAlPS_2.png
  → VDOS analysis completed with prefix: vdos_LiAlPS
VDOS analysis completed.
```