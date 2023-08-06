# GPLearn

## Description
GPLearn is a package written independently for the Cherenkov telescope, based on the python language and the pytorch model, we packaged the different functions into separate modules, and allowed the user to customize the module parameters according to the needs.
The results of a series of data simulations and analyses performed with GPLearn are shown in detail, so that the reader can not only use the data provided by us to reproduce the results shown in the article, but also to analyze the data of the actual detector, and we hope that the results can be widely used in the field of astrophysics.<br><br>

## Update Log
1. stable-1.0.0 (latest version, updated on 2023.07.16 UTC+8)
+ The official version is now available
<br><br>

## Software Architecture
In order for you to use GPLearn properly, make sure you have the following dependencies installed in your current environment:
1. python  ^3.8.5<br>
2. pytorch  ^1.7.0<br>
3. torchvision  ^0.8.0<br>
4. matplotlib  ^3.4.0 <br>
5. numpy  ^1.22.2<br>
6. pandas  ^1.2.4<br><br>

## Installation
1. Run the following clone command at the location you specified:<br><center>**git clone https://gitee.com/chengaoyan/gplearn.git**</center>
2. If you are downloading the program for the first time, first run the **init.py** file located in the program's home directory. This script checks to see if the **/data** directory and its subdirectories in the program's home directory are complete (and creates a patch if they are not).
3. The **main.py** file located in the main directory of the program is a job script where you can set up the execution flow of the program according to your different needs, and you may need to change the **settings.json** file for some parameter adjustments, which will be explained in the following sections. In addition, **main_quick_start.py** provides a simple demo program for background suppression, which you can run directly.
4. We recommend that you use the **nohup** command to host your run on the supercomputing server, for which we have written an adapted logging function and remaining time prediction function, and all your outputs (including all intermediate results) will be automatically saved!
<br><br>

## Instructions
### Preparations
In order to run GPLearn correctly, please make sure that you have python and its dependencies installed on your computer. If you are a first-time user, first run the init.py script in the package root directory, which will automatically check the integrity of your program directory and automatically fill in the missing files. <center> **python init.py** </center>

**/data/origin** contains detector data for photons and protons at different energies. Considering the overall size of the package and the integrity of the program, we have included some of the demo data in the official version, which will make your program work. In addition, you can refer to our data format, modify your data accordingly and use it for GPLearn.
<br>
Our recommended main entry point for the program is the **main.py** file located in the GPLearn root directory, which allows you to customize the workflow according to your needs. In addition, we provide a quick start script that you can run **main_quick_start.py** and make changes based on it.
<br><br>

### Mainstreaming
For convenience, we have integrated all data loading, model creation, training and testing processes into **automation**, which you only need to bring in at the main entry point of the program, **main.py**<br><center>**from bin.automation import Au**</center>

In the initialization of the **Au** class, you need to specify the following:
<!-- <center> -->

| Parameters | Types | Defaults | Descriptions |
| --- | --- | --- | --- |
| gamma_energy_list | list | - | mandatory, indicates the photon energy levels involved in the training (please note that you need to make sure that the **/data/origin** directory contains the appropriate raw data files) |
| proton_energy_list | list | - | Mandatory, indicates the proton energy levels involved in the training |
| particle_number_gamma | int | - | mandatory, specify the number of photons to be loaded for each energy point (please note that you need to make sure that the corresponding raw data file contains enough example data as much as possible, otherwise it may result in the final number of photons loaded to be less than the set number) |
| particle_number_proton | int | - | Mandatory, specify the number of photons to be loaded per energy point |
| allow_pic_number_list | list | - | mandatory, specify the allowed number of excited detectors, the program will load from the raw dataset in the order specified in the list the data of the cases that have triggered a specific number of detectors until the set number is reached or until all the data that meets the condition is loaded | | limit_min_pix | int | - | mandatory, specify the number of photons to load per energy point |
| limit_min_pix_number | bool | False | If or not need to limit the pixel excitation threshold, if True, the program will read the settings.json |
| ignore_head_number | int | 0 | Set dataset bias, specify program load from current position, i.e., will ignore data before this index |
| interval | float | 0.8 | Sets the percentage of data used for training |
| batch_size | int | 1 | Sets the number of samples to be trained in a single run (note that if you set multi-GPU cluster computing in settings.json, make sure this parameter is set to greater than or equal to 2) |
| use_data_type | str | None | Set the type of data to be used for loading (note that you need to make sure that the **/data/origin** directory contains the appropriate raw data file) |
| pic_size | int | 64 | Sets the size of the cropped picture |
| centering | bool | True | Sets whether the image cropping process centers the signal area in the center |
| use_weight | bool | False | Sets whether the coordinates of the center point of the signal area are weighted according to the signal strength in the centering process |
| train_type | str | particle | Sets the type of model training task, currently only background suppression (particle), energy reconstruction (energy), core position reduction (position), and incidence orientation reduction (angle) are supported |
| need_data_info | bool | False | Sets whether to need detailed data information about the training process (different from logs), if True, the program records the loss and correctness of each iteration of the training process and the testing process, as well as the final result, and saves it to **/data/info** |
| use_loading_process | int | None | Specify multi-process loading of data (note that the current parameter is disabled and the program forces single-process loading) |
| current_file_name | str | main.py | Specify the name of the entry file for the main function, if you need to execute the program in another file, change it to the appropriate filename | current_file_name | main.py |

<!-- </center> -->
<br>In addition, **settings.json** contains some settings for running the program, and its usage is described here:<br>
1. The **loading_min_pix** module controls the pixel excitation threshold during image preprocessing. If the parameter **limit_min_pix_number** is set to **True** in the initialization of the **Au** class, the program reads the corresponding setting under the parameter, otherwise it is ignored.
+ The parameter **uniformThreshold** indicates whether to use uniform energy threshold, if it is **true**, the program will use the set **gammaUniform** threshold for photons and **protonUniform** threshold for protons respectively; if it is **false**, the program will read the **gamma* * and **proton** thresholds for photons and protons respectively. **gamma* and **proton** thresholds at different energies will be read (please make sure you set the thresholds for the corresponding energies, otherwise the program will treat them as 0). 2. **GPU*** thresholds will be read for the protons, and **protonUniform** thresholds will be read for the protons.
2. **GPU** module controls the computing device used for deep learning, for general devices, the program will use the main GPU, no additional settings are needed at this time.
+ For devices with multiple GPUs, you can manually set the number of participating devices by changing the **mainGPUIndex** parameter; in addition, you can enable the **multiple** parameter to enable the use of multiple graphics cards to compute at the same time, and the set of graphics card numbers used can be set by the **multipleGPUIndex** parameter. Please note that if you use multiple graphics cards for parallel computing, please make sure that the **batch_size** parameter is greater than or equal to 2 in the initialization of the **Au** class.
<br><br>
The constructor of the class does the data loading work, which can take a long time. When class Au is fully constructed, we can call its built-in function **load_model** to load the model, to which we need to specify parameters:

| Parameters | Types | Defaults | Descriptions |
| --- | --- | --- | --- |
| modelName | str | - | Mandatory, the full name of the model to be called or created (please note that the name does not need to specify a path and the model name needs to end with a .pt suffix) |
| model_type | str | - | Mandatory, model type, commonly used model types: ResNet_custom for background suppression work, GoogLeNet for energy reconstruction tasks, PointNet for core coordinate reduction, and AngleNet for incidence direction reduction (please note that if you are using a customized model, please make sure to follow the prescribed process strictly) | modelInit | bool | False | Specifies whether the model should be initialized, set this parameter to True if you need to create an untrained model parameter. |
| modelInit | bool | False | Specifies whether the model is initialized, set this parameter to True if you need to create an untrained model parameter (note that the system overrides writes if there is a file that is the same as **modelName**) |

In addition, GPLearn supports the use of custom models, but please follow the steps below closely:<br>
1. Make sure your custom model inherits the **torch.nn.Module** class and has a non-duplicate name.
2. You need to add an introduction to the **/model/__init__.py** file to ensure that the main function is accessible.
3. for your custom model to be recognized by the automated training class **Au**, you need to introduce your custom model in the **/bin/modelInit.py** file and add the corresponding branch to the model selector statement.
4. At this point, you can use your custom model as if it were a pre-built model

After the model is loaded successfully, we can carry out training through the class built-in function **train_step**, and we can set the corresponding parameters of the training process independently:

| Parameters | Types | Defaults | Descriptions |
| --- | --- | --- | --- |
| epoch_step_list | list | - | Mandatory, the parameters will be passed as a list indicating the number of repetitions needed for each stage in turn |
| lr_step_list | list | None | A list can be passed to specify the learning rate of the model for each stage (note that the learning rate of the stage should be the same length as the number of times the stage is trained, if not specified, it means that all the stages follow a learning rate of 6e-6) |

<br><br>
Of course, for the testing task, we just need to load the data and test the results with the current model, which can be done using the class's built-in **test** function:<br>
This function does not need to be passed any parameters and will print the test results of the model as it executes.
<br><br>
Finally, at the end of our task, call the class destructor **finish** to save the generated logs and data.
<br><br>
At this point, we have edited the main program in its entirety, and you are ready to deploy your current project on a computing platform. Since deep learning training tends to be long, we recommend using the **nohup** command:<center> **nohup python main.py >> /dev/null 2>&1 &** </center>

This performs computational tasks in the background on your target server.<br>
Please note that for general syntax errors as well as system errors, the **nohup** command may lack the appropriate logging. Therefore, please make sure that your model can run successfully before hosting it.
<br><br>

### Accessibility
Whether you are running in a debugger, **nohup** hosted, or submitting to a compute cluster, we have developed a standalone logging module to assist in **GPLearn** operation: no matter where the program is running, you can access the specified log file to see all the output from the current program run. Our log files are stored in the **/data/log** directory and are named with the timestamp of the execution of the program, so you can easily locate the log file of the current program. With the help of the log, you can:<br>
1. restore the main entry file of the program execution: the log will save all the codes in the entry file of the program, so you can easily restore the codes;
2. restore the main settings of the program execution: the log will save all the settings in **settings.json** that are called in the program execution, you can easily realize settings restoration;
3. view the information about the program running: the log will save the current program version number, running process ID, program main entry path and other information;
4. View training tasks and progress: the log will save all the output information during data processing, model loading, training and testing, you can view the length of the loaded dataset, the model information, the training iteration loss, the results of the test set and the corresponding completion time, which will play an important role in your judgment;
5. View the estimated remaining time: the log will give you the estimated remaining time and the estimated completion time according to the previous program operation, which will serve as the basis for you to reasonably plan the time allocation.

After the training is complete, if you have recorded detailed data information about the training process (usually a **.data** file with the same name as the log file, located in the **/data/info** directory), you can call GPLearn's built-in drawing package <center> **from bin.draw import \*** </center>

We strongly recommend that you execute the drawing part of the code in Jupyter Notepad. The final test result of the program can be shown by the **result_genereation** function, which only needs to be passed the corresponding process data file (**.data** file).<br>
Of course, you can customize the style of the output image by setting the **drawing -> result** section under the **settings.json** file; for different result images of different training tasks, you can make the following settings:
| Keyword | Default | Description |
| ---  | --- | --- |
| title | - | Outputs the title of the chart, which will be used as the file name if Save Image As is set |
| xlabel | - | Horizontal axis title |
| ylabel | - | Vertical Axis Header |
| label | Deep-Learning | graph line title |
| TeV_mode | false | Whether the energy value is in **TeV** |
| logX_mode | false | Whether the horizontal coordinate is logarithmically labeled |
| logY_mode | false | Whether the vertical coordinate is labeled in logarithmic form |
| color | null | graph line color |
| save -> switch | false | whether you need to save as image |
| save -> head_name | null | Save as image header name |
| save -> dpi | 400 | Save as image definition |
