#!/usr/bin/env python3

import json

ensemble = "/glurch/scratch/knd35666/ensembles/ens001"
ensemble_config = "2200.config"

fermion_ps = [
    {
        "csw_r": 1.0,
        "csw_t": 1.0,
        "mass": m,
        "boundary_phases": [1, 1, 1, 1],
        "isAnisotropic": False,
        "xi_0": 1.0,
        "nu": 1.0,
    }
    for m in [
        -0.50,
        -0.51,
        -0.52,
        -0.53,
        -0.54,
        -0.55,
        -0.56,
        -0.57,
        -0.58,
        -0.59,
        -0.6,
    ]
    # for m in [-0.551,-0.552,-0.553,-0.554,-0.555,-0.556,-0.557,-0.558,-0.559,-0.561,-0.562,-0.563,-0.564,-0.565,-0.566,-0.567,-0.568,-0.569]
]

training_epochs = 1000
check_icg_every = 10
adam_lr = 1e-2
batchsize = 1
nr_layers = 8
filter_iteration = 10
lens_space = [1, 2, 4]
lens_time = [1, 2, 4, 8]

out_path = "../"

for i, fermion_p in enumerate(fermion_ps):
    parameters = {
        "ensemble": ensemble,
        "config": ensemble_config,
        "fermion_parameters": fermion_p,
        "training_epochs": training_epochs,
        "check_icg_every": check_icg_every,
        "adam_lr": adam_lr,
        "batchsize": batchsize,
        "nr_layers": nr_layers,
        "out_path": out_path,
        "filter_iteration": filter_iteration,
        "lens_space": lens_space,
        "lens_time": lens_time,
    }

    with open(f"parameters.{i}.json", "w") as fout:
        json.dump(parameters, fout)

print(i)
