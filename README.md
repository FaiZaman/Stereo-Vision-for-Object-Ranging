# Stereo Vision for Object Distancing

This project aims to combine stereo vision with an object ranging algorithm (YOLO) and add in object recognition.

# Setup

Clone the GitHub repository. Warning: the dataset is large (~2GB). Navigate to `stereo_vision_for_object_ranging.py` and edit the `master_path_to_dataset` variable to be a string of the path to the dataset in your system. Change the `skip_forward_file_pattern` to a time in order to skip to that specific time.

The default approach is dense stereo with all its pre and postprocessing and without WLS filtering
	- To switch to WLS Filtering with dense stereo, set the "WLS_on" variable to True						    (line 19)
	- To switch to sparse stereo, set the "sparse_ORB" variable to True								            (line 20)
	- To switch back to dense stereo with no WLS filtering, set both variables to False						    (lines 19, 20)
	- IMPORTANT: DO NOT SET BOTH TO TRUE AT THE SAME TIME!

# Running

Run the program by running `stereo_vision_for_object_ranging.py` either in the command prompt or from an IDE of your choice, and sit back and watch the program recognise objects and distances! 