# Event Type Identification
3D CNN or PointNet to identify the class of an event in an AMEGO-X/ComPair type system.

Needs the following libraries
* MEGAlib (for generating the simulated the data)
* PyTorch
* PyRoot
* Torchmetrics

# Example

To reproduce my current results, you need to do the following

1. Generate simulated data. You can either just ran `cosima` in multiple terminals or run `mcosima` with a number of instances to run. 
    * `cosima -z ~/ComPair/eventtypeidentification/resource/Sim_1MeV_50MeV_flat.source`
    * `mcosima -t 35 -z ~/ComPair/eventtypeidentification/resource/Sim_1MeV_50MeV_flat_AMEGOX.source`
2. Process data to generate a dataset that will be input to the machine learning code. This is done with the `preprocess_sims.py` code
    * `python preprocess_sims.py -path /data/slag2/njmille2/1MeV_50MeV -outfn /data/slag2/njmille2/test_dataset_nhits2_detector1_2500000.pkl -minhits 2 -nevents_dataset 2500000 -dtype 1`
    * The path lists the directory to all the cosima files. The code will find all the cosima files in that directory. It will read in a single file and process it until it has processed the whole file and move on to the next, unless it has found enough valid events in which it will then dump the data to the specified pickle file.
    * The "minhits" option gives the minimum number of hits for an event
    * The "nevents_dataset" gives the number of events to use for each type. For the example, the output dataset will have 5 million events with 2.5 million each of Compton and pair events.
    * dtype requires each accepted event to have started in the corresponding detector type
3. Run the `fit_cnn_model_binary_distributed.py` code. 
    * I have been mainly using the "distributed" version, 
    * `python fit_cnn_model_binary_distributed.py -fn /data/slag2/njmille2/test_dataset_nhits2_detector1_2500000.pkl -dir /data/slag2/njmille2 -label June13 -model TestNet1 -batch 128`
    * This should run on all GPUs on the computer. It is coded for single node / multiple GPU.
    * The "dir" option specifies the output directory for the best model parameters and a text file with some information about loss and accuracy for each epoch.
    * The "label" option specifies a label to be given to each output.
    * The "batch" option is the batch size FOR EACH GPU.
4. Run `fit_pn_model_binary_distributed.py` to run the PointNet version of the code.
    * This runs faster than the CNN model. Therefore, we can much many more epochs.
    * Seems to get slightly higher accuracy.
5. There are partially finished Equinox/Jax versions of the code. I was just trying to learn Equinox/Jax by porting the models and getting it to run on the multiple GPUs. The CNN version should be working, but the PointNet version still has some bugs to fix.

# Modifying the model

The PointNet model is stored in the `models/PointNet.py` code while the CNN is generated in the `models/models.py` code.