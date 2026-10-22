# Exploring Event Camera-based Odometry for Planetary Robots
<p align="center">
    <img src="https://rpg.ifi.uzh.ch/img/papers/arxiv22_Mahlknecht.png" alt="EKLT-VIO" width="600"/>
</p>

This repository contains code that implements the event-based VIO algorithm described in Mahlknecht et al. RAL'22. The paper can be found [here](http://rpg.ifi.uzh.ch/docs/RAL22_Mahlknecht.pdf), and datasets can be found [here](https://uzh-rpg.github.io/eklt-vio/).
If you use this code in an academic context, please cite the following work:

[Florian Mahlknecht](https://florian.world), [Daniel Gehrig](https://danielgehrig18.github.io/), Jeremy Nash, 
Friedrich M. 
Rockenbauer, Benjamin Morrell, Jeff Delaune, [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html), "Exploring Event Camera-based Odometry for Planetary Robots", IEEE Robotics and Automation Letters (RA-L), 2022
```bibtex
@article{Mahlknecht22RAL,
  title={Exploring Event Camera-based Odometry for Planetary Robots},
  author={Mahlknecht, Florian and Gehrig, Daniel and Nash, Jeremy and Rockenbauer, Friedrich M. and Morrell, Benjamin and Delaune, Jeff and Scaramuzza, Davide},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2022}
}
```

## x_evaluate

Performance evaluation for (range-) visual-inertial odometry xVIO framework.

Provides an extensive toolset to evaluate frontend and backend accuracy (i.e. pose and feature tracking) as well as 
computational efficiency, such as realtime factor, CPU and memory usage.

### Quick start with Docker (any Linux host)

ROS Noetic is only available for Ubuntu 20.04, so on newer hosts use the provided container. It ships ROS Noetic, the
X library with event-based frontends ([jpl-x/x_events](https://github.com/jpl-x/x_events)), all catkin dependencies 
and the Python requirements. This repository is mounted into the container, so edits on the host are picked up directly.

The reference data is the [Event Camera Dataset](https://rpg.ifi.uzh.ch/davis_data.html) (DAVIS 240C: events, frames,
IMU and motion-capture ground truth), on which the paper's EKLT-VIO results (Table II) are reproduced. Download the 10
sequences (about 11.7 GB) into `../data/rpg_davis_data/` on the host:

```bash
mkdir -p ../data/rpg_davis_data && cd ../data/rpg_davis_data
for s in boxes_6dof boxes_translation dynamic_6dof dynamic_translation hdr_boxes hdr_poster \
         poster_6dof poster_translation shapes_6dof shapes_translation; do
  wget -c https://download.ifi.uzh.ch/rpg/web/datasets/davis/$s.bag
done
```

Then build and evaluate:

```bash
docker/run.sh                       # first run builds the image (~15-20 min, ceres is compiled from source)
# inside the container:
catkin build x_evaluate && source devel/setup.bash
python3 src/x_evaluate/test/evaluate.py --configuration src/x_evaluate/test/evaluate_rpg_table2.yaml \
  --output_folder /data/rpg_runs/eklt --dataset_dir /data --frontend EKLT --name "EKLT"
```

The sequences run one after the other (about 2 h for EKLT on a laptop; `--frontend XVIO` takes minutes). Each one writes
a subfolder with plots and `trajectory_tracking_summary.xlsx`; the expected mean position errors are listed in
`docs/eklt_vio_reproduction_notes.md` (section 7, average 0.54%). To run a single sequence, copy its entry from
`test/evaluate_rpg_table2.yaml` into a YAML file of its own.

Datasets are read from `../data` (override with `XVIO_DATASET_DIR=/path/to/datasets docker/run.sh`) and mounted at
`/data`. To modify the X library itself, clone `x_events` (branch `events`) on the host, apply
`docker/patches/*.patch` with `git apply`, and start with `X_SRC=/path/to/x_events docker/run.sh`.

Notes:
- `docker/patches/x_events-clamp-feature-tile.patch` fixes a segfault in `x_events` when an EKLT feature drifts 
  slightly outside the image (its tile row underflowed and indexed past a per-tile array).
- `evaluate.py` does not check whether the C++ `evaluate` crashed; check each run's console output.
- Most Wells Cave configs (`test/evaluate_wells*.yaml`) expect `/dvs/image_reconstructions`, E2VID images that were
  not published; on the public raw frames (`/dvs/image_raw`) EKLT-VIO mostly diverges. See
  `docs/eklt_vio_reproduction_notes.md` for details.

### Dependencies

The library has been tested only on Ubuntu 20.04 with Ros Noetic. Although no roscore is needed, feature tracking 
evaluation requires Python3 rosbag reader, therefore lifting the requirements to Ros Noetic.

Beyond ROS, the major dependencies are the X library and gflags_catkin (see `dependencies.yaml`). Typical installation
commands might be:

```bash
cd YOUR_ROS_WORKSPACE/
source devel/setup.zsh  # or .bash if you use BASH
cd src
git clone ... x_evaluate
vcs-import < x_evaluate/dependencies.yaml
catkin build x_evaluate  # DO NOT BUILD all packages, as the rpg_ros_driver might fail, but it's not needed
```

For the python package it is recommended to create a virtual environment and install the requirements with pip:

```bash
conda create -n x python=3.8
conda activate x
pip install -r requirements.txt
pip install . # Optionally install the x_evaluate in the python dist-packages
```

### Basic architecture

The library consists of one C++ file which directly calls X library callbacks such as `processIMU()` from the data
read from a rosbag. In this way a runtime-independent evaluation can be performed. Results are then dumped to CSV files 
and analyzed in Python, with the main module `x_evaluate`.

The python evaluation runs additionally dump all the main results to a pickle file, such that different runs can be 
easily compared and reproduced.

### Usage

An evaluation run is performed by `evaluate.py`. A configuration file e.g. `evaluate.yaml` defines which datasets are
processed and based on which base parameter files, e.g. `params_rpg_davis.yaml`. Parameters can be overwritten for a  
single sequence or for all sequences on the respective entry in `evaluate.yaml`. Additionally,  by passing
`--overrides some_param=17 some_other_param=False` out-rules all other settings for the purpose of parameter tuning runs.

A simple example run:

```bash
python test/evaluate.py --configuration test/evaluate.yaml --output_folder /path/to/_out/ \
--dataset_dir /path/to_datasets --frontend XVIO --name "XVIO"
```

And a comparison run:

```bash
python scripts/compare.py --input_folder /path/to --sub_folders _out:_out_2 ----output_folder /path/to/results
```


