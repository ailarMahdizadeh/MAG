# MAHGNN

1) Downloading data

You need to download the functional connectivity matrices obtained from cc200 and cc400 atlases.

https://drive.google.com/file/d/1HlN1pRhq516GUZv8e3ilyyu4pUBzD1U-/view?usp=sharing


Note: The FCN files are gained directly from BrainGNN but they are arranged in single folders related to number of ROIS. 

Indeed this step was essential because there were subject IDS in csv file that does not have file but their info was present in csv file related to subjects info.
So, firstly I cleaned the csv file by deleting the IDS that in front of them was written 'no_file', and based on new updated subject_info table, I extract the correlation matrices corresponding to each ROI.

The matlab code that I wrote for this arrangement is available in

https://drive.google.com/file/d/1ghoSJVKNwDftuv_9XAO2lIqrrst37V0v/view?usp=sharing



2) Running

You need to run Training.py 
