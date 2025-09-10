import argparse
import collections
import pickle
import time
import random
import os
import glob

import numpy as np

# import matplotlib.pyplot as plt

import ROOT as M

def get_bin_idx(nhits):
    # First attempt bins of 2-10, 11-20, 21-40, and >40
    
    if nhits <= 10:
        idx = 0
    elif nhits > 10 and nhits <= 20:
        idx = 1
    elif nhits > 20 and nhits <= 40:
        idx = 2
    elif nhits > 40:
        idx = 3

    return idx

def load_data(fn, event_hits, maxevents, minhits=1, maxhits=None, detectortype=None, maxclass=None,
              geometry="amegox"):
    """
    Prepare numpy array datasets for scikit-learn and tensorflow models

    Returns:
        list: list of the events types in numerical form: 1x: Compton event, 2x pair event, with x the detector (0: passive material, 1: tracker, 2: absober)
        list: list of all hits as a numpy array containing (x, y, z, energy) as row 
    """

    print("{}: Load data from sim file".format(time.time()))

    # Load MEGAlib into ROOT
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

    # Initialize MEGAlib
    G = M.MGlobal()
    G.Initialize()

    finished = False

    # Fixed for the time being
    geometry = "amegox"
    if geometry == "amegox":
        GeometryName = "~/ComPair/Geometry/AMEGO_Midex/AmegoXBase.geo.setup"
    elif geometry == "grips":
        GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"
    elif geometry == "gripsrealistic":
        GeometryName = "~/ComPair/eventtypeidentification/resource/GRIPS_realistic.geo.setup"

    # Load geometry:
    Geometry = M.MDGeometryQuest()
    if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
        print("Geometry " + GeometryName + " loaded!")
    else:
        print("Unable to load geometry " + GeometryName + " - Aborting!")
        quit()

    Reader = M.MFileEventsSim(Geometry)
    if Reader.Open(M.MString(fn)) == False:
        print("Unable to open file " + fn + ". Aborting!")
        quit()
    
    #Hist = M.TH2D("Energy", "Energy", 100, 0, 600, 100, 0, 600)
    #Hist.SetXTitle("Input energy [keV]")
    #Hist.SetYTitle("Measured energy [keV]")


    # First attempt bins of 2-10, 11-20, 21-40, and >40


    event_types = []
    nhits_list = []

    max_label = 0
    NEvents = 0
    i_tmp = 0
    while True: 
        Event = Reader.GetNextEvent()
        M.SetOwnership(Event, True)
        if not Event:
            break
        i_tmp += 1
        if i_tmp % 40000 == 0:
            tmp = [len(event_hits[ii]) for ii in event_hits]
            print("Processing event", i_tmp, tmp)

        # Type = 0
        if (detectortype is not None) and Event.GetIAAt(1).GetDetectorType() != detectortype:
            continue

        Type = None
        if Event.GetNIAs() > 0:
            if Event.GetIAAt(1).GetProcess() == M.MString("COMP"):
                Type = 0# + Event.GetIAAt(1).GetDetectorType()
            elif Event.GetIAAt(1).GetProcess() == M.MString("PAIR"):
                Type = 1#0 + Event.GetIAAt(1).GetDetectorType()
        else:
            # Why break???
            break

        if Type is None:
            continue


        if Type+1 > max_label:
            max_label = Type + 1

        nhits = Event.GetNHTs()
        
        # Reject if too little number of hits
        if nhits < minhits:
            continue

        # Reject Compton if it has too many hits
        if (maxhits is not None) and (Type == 0) and (nhits > maxhits):
            continue

        haspair = False
        if Type == 0:
            nias = Event.GetNIAs()
            for i in range(2, nias):
                if Event.GetIAAt(i).GetProcess() == M.MString("PAIR") and Event.GetIAAt(i).GetDetectorType() == 1:
                    haspair = True
                    Type = 2
                    break

        if haspair:
            continue
        
        idx_hits = get_bin_idx(nhits)
        idx_eh = 10*Type + idx_hits
        
        if (maxclass != None) and (len(event_hits[idx_eh]) >= maxclass):
            continue

        nhits_list.append(nhits)

        Hits = np.zeros((Event.GetNHTs(), 4))

        for i in range(0, Event.GetNHTs()):
            Hits[i, 0] = Event.GetHTAt(i).GetPosition().X()
            Hits[i, 1] = Event.GetHTAt(i).GetPosition().Y()
            Hits[i, 2] = Event.GetHTAt(i).GetPosition().Z()
            Hits[i, 3] = Event.GetHTAt(i).GetEnergy()

        NEvents += 1
        
        if idx_eh not in event_hits:
            print("Adding:", idx_eh)
            event_hits[idx_eh] = []

        event_hits[idx_eh].append(Hits)

        # event_types.append(Type)
        event_types.append(idx_eh)

        if (maxevents != None) and (NEvents >= maxevents):
            break

        # len_classes = [len(event_hits[i]) for i in event_hits]
        # if (maxclass != None) and (len(event_hits[0]) >= maxclass) and (len(event_hits[1]) >= maxclass):
        hitmax = [len(event_hits[key]) >= maxclass for key in event_hits]
        if (maxclass != None) and all(hitmax):
            print("Breaking on max number of events for each class")
            for key in event_hits:
                print(key, ":", len(event_hits[key]))
            # print("0:", len(event_hits[0]))
            # print("1:", len(event_hits[1]))
            finished = True
            break

    
    print("Total number processed = ", i_tmp)

    print("Occurrences of different event types:")
    print(collections.Counter(event_types))
    print("Occurances of Nhits:")
    count_nhits = collections.Counter(nhits_list)
    print(count_nhits)
    nhits_array = np.array(nhits_list)

    print("Max Label:", max_label)

    # return event_hits, finished
    return finished

def gen_dataset(event_hits, nevents, split):
    """This justs takes a random subset of each event type so that we have
    similar number of events of each type. It returns a list of the hits and types
    for the dataset."""

    event_types = np.array(list(event_hits.keys()))

    for ut in event_types:
        events = event_hits[ut]
        num = len(events)
        if num < nevents:
            print("Reducing nevents from", nevents, "to", num)
            nevents = num

    dataset_hits = []
    dataset_types = []
    for ut in event_types:
        events = event_hits[ut]
        num = len(events)
        print("Number of type", ut, "is", num)
        if num < nevents:
            print("Don't have enough events of type", ut, num)
            print("Using all events of this type in dataset")
        else:
            random.shuffle(events)
            events = events[:nevents]
        types = ut*np.ones(len(events), dtype=int)

        dataset_hits = dataset_hits + events
        dataset_types = dataset_types + list(types)
    
    return dataset_hits, dataset_types

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Cosima Sims')

    # parser.add_argument('-fn', dest='fn', action='store', help='Filename to parse')
    parser.add_argument('-path', dest='path', action='store', help='Path to files')
    parser.add_argument('-nevents', dest='nevents', type=int, action='store', help='Number of events')
    parser.add_argument('-minhits', dest='minhits', type=int, action='store', default=1,
                         help='Mininum number of hits for good event.')
    parser.add_argument('-maxhits', dest='maxhits', type=int, action='store', default=None,
                         help='Maximum number of hits for Compton events.')
    parser.add_argument('-outfn', dest='outfn', action='store', help='Path to store output file.')
    # parser.add_argument('-outpath', dest='outpath', action='store', help='Path to output train/val datasets')
    parser.add_argument('-preprocess_only', dest='preprocess_only', action='store_true')
    parser.add_argument('-nevents_dataset', dest='nevents_dataset', type=int, action='store', default=500000)
    parser.add_argument('-dtype', dest='dtype', type=int, action='store', help='Detector type number')
    parser.add_argument('-split', dest='split', type=float, action='store', help='Training/Validation split')
    parser.add_argument("-geometry", dest="geometry", action='store', help="Name of geometry")

    args = parser.parse_args()

    # 1. Load the Cosima sim data, do some filtering, and separate into different event types
    # pattern_glob = os.path.join(args.path, "*.sim.gz")
    pattern_glob = args.path + "/*.id1.sim.gz"
    fns = glob.glob(pattern_glob)
    print("fns:", fns)

    event_hits = {}
    # event_hits[0] = []
    # event_hits[1] = []
    event_hits[0] = []
    event_hits[1] = []
    event_hits[2] = []
    event_hits[3] = []
    event_hits[10] = []
    event_hits[11] = []
    event_hits[12] = []
    event_hits[13] = []

    maxclass = args.nevents_dataset
    for fn in fns:
        print("Processing:", fn, len(event_hits[0]), len(event_hits[1]))
        is_finished = load_data(fn, event_hits, args.nevents, minhits=args.minhits, maxhits=args.maxhits,
                                maxclass=args.nevents_dataset, detectortype=args.dtype)
        if is_finished:
            break

    if args.preprocess_only:
        if args.outfn is not None:
            with open(args.outfn, 'wb') as f:
                pickle.dump(event_hits, f)
        exit()

    #2. Get similar number of each event type and combine to form a dataset
    dataset = gen_dataset(event_hits, args.nevents_dataset, args.split)

    print("Size of dataset:", len(dataset[0]))
    if args.outfn is not None:
        with open(args.outfn, 'wb') as f:
            pickle.dump(dataset, f)