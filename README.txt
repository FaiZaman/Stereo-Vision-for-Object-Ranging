STEREO VISION FOR OBJECT RANGING

To run:

1) Open "stereo_vision_for_object_ranging.py" in an editor
2) Edit the "master_path_to_dataset" variable if necessary 										    (line 12)
3) The University machines add "._" to the beginning of the image filenames for some reason 
	- If your machine does not do this, remove the lines where "filename[2:len(filename)]" appears for filename_left and filename_right (lines 159, 160)
4) The default approach is dense stereo with all its pre and postprocessing and without WLS filtering
	- To switch to WLS Filtering with dense stereo, set the "WLS_on" variable to True						    (line 19)
	- To switch to sparse stereo, set the "sparse_ORB" variable to True								    (line 20)
	- To switch back to dense stereo with no WLS filtering, set both variables to False						    (lines 19, 20)
	- IMPORTANT: DO NOT SET BOTH TO TRUE AT THE SAME TIME!
5) Run the file
