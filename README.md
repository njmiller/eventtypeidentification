# Event Type Identification
3D CNN to identify the class of an event in an AMEGO-X/ComPair type system.

Needs the following libraries
* MEGAlib (for generating the simulated the data)
* PyTorch
* PyRoot
* Scikit-Learn
* PyTorch Lightning (if running the lightning script)

# Example

To reproduce my current results, you need to do the following

1. Generate simulated data. Currently I just ran `cosima` in multiple terminals to make it go faster. It is probably better to use the parallel version of cosima, but I had already started 
    * `cosima -z ~/ComPair/eventtypeidentification/resource/Sim_1MeV_50MeV_flat.source`
2. Process data to generate a dataset that will be input to the machine learning code. This is done with the `preprocess_sims.py` code
    * `python preprocess_sims.py -path /data/slag2/njmille2/1MeV_50MeV -outfn /data/slag2/njmille2/test_dataset_nhits2_detector1_2500000.pkl -minhits 2 -nevents_dataset 2500000 -dtype 1`
    * The path lists the directory to all the cosima files. The code will find all the cosima files in that directory. It will read in a single file and process it until it has processed the whole file and move on to the next, unless it has found enough valid events in which it will then dump the data to the specified pickle file.
    * The "minhits" option gives the minimum number of hits for an event
    * The "nevents_dataset" gives the number of events to use for each type. For the example, the output dataset will have 5 million events with 2.5 million each of Compton and pair events.
    * dtype requires each accepted event to have started in the corresponding detector type
3. Run the `fit_cnn_model_binary_XXX.py` code. There are two versions of the code that I can use right now, the "distributed" and the "lightning" code. 
    * I have been mainly using the "distributed" version, but you should know that the accuracy listed after each epoch probably needs to be multiplied by the number of GPUs as each process will only test N / N_gpus but calculate an accuracy based on testing N. I am working on fixing this.
    * `python fit_cnn_model_binary_distributed.py -fn /data/slag2/njmille2/test_dataset_nhits2_detector1_2500000.pkl -dir /data/slag2/njmille2 -label June13 -model TestNet1 -batch 128`
    * This should run on all GPUs on the computer. It is coded for single node / multiple GPU.
    * The "dir" option specifies the output directory for the best model parameters and a text file with some information about loss and accuracy for each epoch.
    * The "label" option specifies a label to be given to each output.
    * The "batch" option is the batch size FOR EACH GPU.

# Modifying the model

Right now, you must modify the model differently for the "distributed" code and the "lightning" code. I plan on combining the two by, most likely, having a separate function that returns lists of layers and the class just executes the layers in order, but I haven't tested to make sure it works yet.

1. For the "distributed" code, all the models are contained in the original "fit_cnn_model_binary.py" file. You can modify the "TestNet1" class to change the model that is run when specifying that model from the commandline.

2. For the "lightning" code, you must modify the "EventTypeIdentification" class.