# User experiments

## Classifier training

### Labels
For the drill assembly task, we created six activity labels:
- `attach_shell`: User attaching the shell to the drill.
- `screw`: User screwing the screws into the shell.
- `attach_battery`: User attaching the battery to the drill.
- `place_drill`: User placing the finished drill on the table.
- `hold`: User being idle.
- `grab_parts`: User grabbing parts given by the robot.

### Installation
- Install pytorch from [here](https://pytorch.org/get-started/locally/).
- Install the required packages using `pip install -r requirements.txt` or `conda install --file requirements.txt`.

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