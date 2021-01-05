## Get started
- First, you should download PR2 mesh files from using [gdown](https://github.com/wkentaro/gdown).
- On grasping_demo folder layer, execute below. You can get the meshes folder in the pr2_description folder.
```
pip install gdown
python scripts/script.py
```

## Dual Grasp learning

First, you should make some folders on the directory of `grasping_demo`.
```
- /grasping_demo
    - /data
        - /rgb_data
        - /depth_data
        - /robot_data
        - /robot_state_data
        - /trained_encoder_model
        - /trained_lstm_model
        - /video
        - /loss
```

When collecting simulation data, execute below.
```
usage:  python pr2_dual_push_scoop_grasp_sim.py  [-n]
                                                 [-g]

optional arguments:
  --try_num, -n    simulation loop times (default: 3)
  --gui, -g        enable gui (default: 1)
```

Exsample:
```
$ python pr2_dual_push_scoop_grasp_sim.py -n 100 -g 1
```

When training, execute below. 
```
usage: python pr2_dual_push_scoop_grasp_train_imgstate2action.py [-p]
                                                                 [-n]
                                                                 [-e]

optional arguments:
  --data_path -p     set datetime of loading data file (default: buffer)
  --training_size -n set train dataset size. it should be same as 
                     try_num at one of the optional arguments of 
                     executing `python pr2_dual_push_scoop_grasp_sim.py` 
                     (default: 100)
  --epochs_num -e    set train epochs size (default: 100)
```

Example:
```
$ python pr2_dual_push_scoop_grasp_train_imgstate2action.py -p 20210101_180627 -n 100 -e 100
```

When inferencing, execute below.
```
usage: python pr2_dual_push_scoop_grasp_test_imgstate2action.py [-p]

optional arguments:
  --model_path -p     set datetime of loading model data file (default: model.pth)
```

Example:
```
$ python pr2_dual_push_scoop_grasp_train_imgstate2action.py -p 20210101_180627 
```

## models
The dish model is downloaded from [here](https://creazilla.com/nodes/71885-candy-dish-3d-model), then number of vertices is reduced and rescaled (x0.001) using meshlab.
The gripper model is Fetch's gripper, urdf of which is modified so that it's base has free 3dof planar joint.
