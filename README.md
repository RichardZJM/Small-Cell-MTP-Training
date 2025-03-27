# Small-Cell MTP Training  

This repository provides example scripts for performing small-cell active learning with Moment Tensor Potentials (MTP) **MLIP-3** and **Quantum Espresso 6.6**. These scripts serve as a reference for users implementing MTP active learning (AL) and small-cell training with Quantum Espresso, but are not intended as plug-and-play solutions.  

Sample K and NaK potentials are provided with their training set in `samplePots`.
## Overview  

Two primary versions of the protocol are available:  
- **`main` branch**: Runs on a fixed allocation of cores, using a primitive semaphore to schedule DFT jobs if core usage exceeds availability (memory usage is unmanaged).  
- **`floatingAllocation` branch**: Uses a dynamic resource allocation, where MD and DFT tasks run as separate SLURM jobs. This requires a cluster with low queue times, tolerant policies, and minimal node launch failures to avoid deadlock.  

## Repository Structure  

Key subfolders include:  
- **`templates`**: Contains constant parameters for task generation.  
- **`io`**: Handles file reading/writing and format conversions.  
- **`activeLearningSections`**: Manages file transfers and software calls.  
- **`ensembles`**: Generates and manages MTP ensembles.  

These components work together in `activeLearnPotential.py` to automate the small-cell training protocol based on a configuration JSON file. An example is provided.

### Data Conversion  

Since Quantum Espresso is used instead of VASP, conversion is handled in the `io` modules via Python dictionaries. Key processing steps include:  
1. **Energy shifting**: Adjusts DFT energies so zero per-atom energy corresponds to an isolated atom.  
2. **Virial stress multiplication**: Multiplies virial stresses by cell volume for MLIP compatibility.  
3. **Index conversion**: Switches between 0- and 1-based type indexing as needed.  

All units are converted to eV and Å.  

## Output  

Running `runActiveLearningScheme` in `activeLearnPotential.py` follows the methodology described in the accompanying paper. The final output includes:  
- The trained potential.  
- The training dataset in MLIP-3 format.  
- All DFT jobs.  
- A log file with active learning metrics.  

The `ensembles` module can generate an ensemble from a completed training set for configuration predictions (but not MD simulations).  

## Installation  

Clone the repository:  
```sh
git clone https://github.com/RichardZJM/Small-Cell-MTP-Training.git
```  

For development (recommended), install in editable mode:  
```sh
pip install -e /path/to/package/
```  

To switch to the floating allocation implementation:  
```sh
cd /path/to/package/
git checkout floatingAllocation
```  

### Usage Example  

```python
from SmallCellMTPTraining.activeLearnPotential import runActiveLearningScheme
import json

with open("/path/to/config.json", "r") as f:
    config = json.load(f)
    runActiveLearningScheme(
        "path/to/training/folder",
        config,
        mtpLevel="12",  # Default is "08"
        initial_train="path/to/initial/training/set",
        initial_pot="path/to/initial/potential",
    )
```  

#### Notes:  
- If no initial training set is provided, `generateInitialDataset` (in `qe.py`) creates one using hydrostatic strains.  
- If an initial training set is provided but no potential, a new potential is trained first.  
- Providing an initial potential **without** an initial training set is invalid.
- There are rudimentary checks to resume terminated runs.

## Customization  

Key functions to adapt for your use case:  
- **`qe.py`** → `generateInitialDataset`: Currently generates hydrostatic strains with fixed DFT parameters.  
- **`md.py`** → `performParallelMDRuns`: Currently launches NPT jobs with random temperatures/pressures.  
- **`writers.py`** → `writeMDInput`: Currently writes single atoms in a randomized BCC lattice.  
- **`templates.py`**: Contains MD/DFT parameters optimized for potassium and sodium.  
