# User experiments

## Classifier training
Instructions for training the activity classifier.

### Labels
For the drill assembly task, we created six activity labels:
- `attach_shell`: User attaching the shell to the drill.
- `screw`: User screwing the screws into the shell.
- `attach_battery`: User attaching the battery to the drill.
- `place_drill`: User placing the finished drill on the table.
- `hold`: User being idle.
- `grab_parts`: User grabbing parts given by the robot.

### Setup
- Install Python 3.8 or higher from [here](https://www.python.org/downloads/).
- Install pytorch from [here](https://pytorch.org/get-started/locally/).
- Install the required packages using `pip install -r requirements.txt` or `conda install --file requirements.txt`.
- Install opencv using `pip install opencv-python`.

### Dataset
The dataset for training the activity classifier consists of user skeletons annotated with their corresponding activity. Download the dataset from this link: [dataset](https://www.dropbox.com/scl/fo/cqb8toktjnmidtg99fzmn/AFbFgfGcpfV1Q-L9kT6uG1Y?rlkey=0nnh24191bynga1azppoe20vj&st=iddt2vuk&dl=0). Save the dataset in a directory under `classifier_training` called `dataset`.

To collect your own dataset, please use a camera with inbuilt body tracking, such as the [Azure Kinect](https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/docs/usage.md) to collect user skeleton data as they perform the actions in this task and manually label each user's data as `XXX-annotated.csv`.

### Training and Testing
To train the classifier, run the following command:
```
python train.py -model_config model_config.yaml
```

To test the classifier, update the `Test` section of `model_config.yaml` with the path to the mean pelvis value and desired test checkpoint. Run the following command:
```
python test.py -model_config model_config.yaml
```

## User study execution
The `ros` directory contains the code used for the user study execution. Ros drivers for the Azure Kinect camera ,Kinova JACO robot arm, and tabletop segmentation and object detection are marked as git submodules.

ToDo: Clean up redundant files.

### Setup
Copy the contents of the `ros` directory to your catkin workspace and build the workspace using `catkin_make` or `catkin build`. Note that some of the packages are interdependent, so you may need to build the workspace multiple times. 

### Launch
To launch the user experiment:
- Launch the necessary drivers: `roslaunch comanip_config adaptive_comanip.launch`
- For the UHTP algorithm: `roslaunch comanip_htn run_demo.launch user_id:=<user_id> mode:=adaptive`
- For the baseline algorithm: `roslaunch comanip_htn run_demo.launch user_id:=<user_id> mode:=fixed`

Instead of the full experiment, there is a demo mode that can be launched using `roslaunch comanip_htn run_demo.launch`.
