from scipy.stats import linregress 
from scipy.stats import norm 
import numpy as np
# Function to find terms within a time range
def find_terms(array, first_value, last_value):
    first_term = next((i for i, x in enumerate(array) if x >= first_value), None)
    last_term = max((i for i, x in enumerate(array) if x <= last_value), default=None)
    return first_term, last_term

# Function to segment data
def segmenter_func(first_term, last, pos_full, dt_full, time_full, cell_full, ke_elec_full, 
                   cell_temp_full, ion_temp_full, tot_energy_full, enthalpy_full, 
                   tot_energy_ke_ion_full, tot_energy_ke_ion_ke_elec_full, vol_full, pressure_full):
    last_term = int(last / 0.7) if last <= int(0.7 * len(pos_full[0, :, 0])) else int(last)
    pos = pos_full[:, first_term:last_term, :]
    step_counts = len(pos[0, :, 0])
    dt = dt_full[first_term:last_term - 1]
    time = time_full[first_term:last_term]
    cell = cell_full[first_term:last_term, :]
    ke_elec = ke_elec_full[first_term:last_term]
    cell_temp = cell_temp_full[first_term:last_term]
    ion_temp = ion_temp_full[first_term:last_term]
    tot_energy = tot_energy_full[first_term:last_term]
    enthalpy = enthalpy_full[first_term:last_term]
    tot_energy_ke_ion = tot_energy_ke_ion_full[first_term:last_term]
    tot_energy_ke_ion_ke_elec = tot_energy_ke_ion_ke_elec_full[first_term:last_term]
    vol = vol_full[first_term:last_term]
    pressure = pressure_full[first_term:last_term]
    return (pos, step_counts, dt, time, cell, ke_elec, cell_temp, ion_temp, 
            tot_energy, enthalpy, tot_energy_ke_ion, tot_energy_ke_ion_ke_elec, vol, pressure)

# Function to evaluate data for diffusivity
def data_evaluator(diffusivity_direction_choices, target_elements, pos, total_ion_array, steps):
    position_data_list, drifted_rectified_structure_list = [], []
    conductor_indices_list, framework_indices_list = [], []
    framework_pos_list, mobile_pos_list = [], []
    mobile_drifted_rectified_structure_list, framework_drifted_rectified_structure_list = [], []
    
    for direction in diffusivity_direction_choices:
        position_data = np.zeros((len(total_ion_array), steps, 3))
        if direction == "XYZ":
            position_data = pos
        elif direction == "XY":
            position_data[:, :, 0:2] = pos[:, :, 0:2]
        elif direction == "YZ":
            position_data[:, :, 1:3] = pos[:, :, 1:3]
        elif direction == "ZX":
            position_data[:, :, [0, 2]] = pos[:, :, [0, 2]]
        elif direction == "X":
            position_data[:, :, 0] = pos[:, :, 0]
        elif direction == "Y":
            position_data[:, :, 1] = pos[:, :, 1]
        elif direction == "Z":
            position_data[:, :, 2] = pos[:, :, 2]
        
        disp = position_data - position_data[:, [0], :]
        conductor_indices = [i for i, element in enumerate(total_ion_array) if element in target_elements]
        framework_indices = [i for i in range(len(total_ion_array)) if i not in conductor_indices]
        
        framework_disp = disp[framework_indices, :, :]
        framework_pos = position_data[framework_indices, :, :]
        mobile_pos = position_data[conductor_indices, :, :]
        
        drift = np.average(framework_disp, axis=0)
        corrected_displacements = disp - drift
        drifted_rectified_structure = position_data[:, [0], :] + corrected_displacements
        
        mobile_drifted_rectified_structure = drifted_rectified_structure[conductor_indices, :, :]
        framework_drifted_rectified_structure = drifted_rectified_structure[framework_indices, :, :]
        
        position_data_list.append(position_data)
        drifted_rectified_structure_list.append(drifted_rectified_structure)
        conductor_indices_list.append(conductor_indices)
        framework_indices_list.append(framework_indices)
        framework_pos_list.append(framework_pos)
        mobile_pos_list.append(mobile_pos)
        mobile_drifted_rectified_structure_list.append(mobile_drifted_rectified_structure)
        framework_drifted_rectified_structure_list.append(framework_drifted_rectified_structure)
    
    return (np.array(position_data_list), np.array(drifted_rectified_structure_list), 
            np.array(conductor_indices_list, dtype=object), np.array(framework_indices_list, dtype=object), 
            np.array(framework_pos_list), np.array(mobile_pos_list), 
            np.array(mobile_drifted_rectified_structure_list), np.array(framework_drifted_rectified_structure_list))