# EXPODE: EXploiting POlicy Discrepancy for Efficient Exploration in Multi-agent Reinforcement Learning

## Install experimental platform

Set up StarCraft II and SMAC:

```shell
cd EXPODE-master/pymarl
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over. We use a modified version of SMAC, which can be installed by following commands:

```shell
cd EXPODE-master/QPLEX_smac_env
pip install -e .
```

The `requirements.txt` file can be used to install the necessary packages into a virtual python environment.

## Run an experiment 
In this code, We evaluate our method on two environments: Predator and Prey("pred_prey_punish"), SMAC("sc2"). We use the default settings in SMAC, and the results in our paper use Version SC2.4.6.2.69232.




To train EXPODE on Predator and Prey, run the following command:
```shell
cd EXPODE-master/pymarl
python3 src/main.py --config=EXPODE_toygame --env-config=pred_prey_punish with env_args.map_name=origin 
```


To train EXPODE on SC2 setting tasks, run the following command:
```shell
python3 src/main.py --config=EXPODE_sc2 --env-config=sc2 with env_args.map_name=MMM2 
```

Map names for SMAC include 2s3z, 3s5z, 5m_vs_6m, 3s5z_vs_3s6z, MMM2, 3s_vs_5z, 6h_vs_8z, corridor.



The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`



Our experimental results are saved in the `EXPODE_experimental_data` folder, and can be found by the following command:

```SHE
cd EXPODE-master/pymarl/src/EXPODE_experimental_data
```

Then, the figures will be shown by running python script:

```shell
python test_plot_sc2.py
python test_plot_stag.py
```

