# emgkit
A toolkit for the processing, analysis and visualization of EMG signals.

## Environment setup
The code is compatible with Python 3.7+. To create and activate the Python environment, run the following commands:
```
python -m venv <ENV_NAME>
source <ENV_NAME>/bin/activate
```

Then, **from within the virtual environment**, the required packages can be installed with the following command:
```
pip install -r requirements.txt
```

## Authors
This work was realized mainly at the [Energy-Efficient Embedded Systems Laboratory (EEES Lab)](https://dei.unibo.it/it/ricerca/laboratori-di-ricerca/eees) 
of University of Bologna (Italy) by [Mattia Orlandi](https://www.unibo.it/sitoweb/mattia.orlandi3/en).

## Citation
If you would like to reference the project, please cite the following paper:
```
@ARTICLE{10552147,
  author={Orlandi, Mattia and Rapa, Pierangelo Maria and Zanghieri, Marcello and Frey, Sebastian and Kartsch, Victor and Benini, Luca and Benatti, Simone},
  journal={IEEE Transactions on Biomedical Circuits and Systems}, 
  title={Real-Time Motor Unit Tracking From sEMG Signals With Adaptive ICA on a Parallel Ultra-Low Power Processor}, 
  year={2024},
  volume={18},
  number={4},
  pages={771-782},
  keywords={Electrodes;Real-time systems;Muscles;Motors;Electromyography;Circuits and systems;Graphical user interfaces;Blind source separation;human-machine interfaces;independent component analysis;low-power;machine learning;on-device learning;online learning;PULP;surface EMG},
  doi={10.1109/TBCAS.2024.3410840}}
```

## License
All files are released under the Apache-2.0 license (see [`LICENSE`](https://github.com/pulp-bio/emgkit/blob/main/LICENSE)).
