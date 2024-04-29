import argparse
import collections
import pickle
import time
import random

import numpy as np

# import matplotlib.pyplot as plt

import ROOT as M

def load_data(fn, maxevents, minhits=1, maxclass=None):
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

    # Fixed for the time being
    GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"

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


    event_types = []
    nhits_list = []

    event_hits = {}
    event_hits[0] = []
    event_hits[1] = []

    max_label = 0
    NEvents = 0
    i_tmp = 0
    while True: 
        Event = Reader.GetNextEvent()
        if not Event:
            break
        i_tmp += 1
        if i_tmp % 40000 == 0:
            print("Processing event", i_tmp, len(event_hits[0]), len(event_hits[1]))

        Type = 0
        if Event.GetNIAs() > 0:
            if Event.GetIAAt(1).GetProcess() == M.MString("COMP"):
                Type += 0# + Event.GetIAAt(1).GetDetectorType()
            elif Event.GetIAAt(1).GetProcess() == M.MString("PAIR"):
                Type += 1#0 + Event.GetIAAt(1).GetDetectorType()
        else:
            break

        if Type+1 > max_label:
            max_label = Type + 1

        nhits = Event.GetNHTs()
        if nhits < minhits:
            continue
        if (maxclass != None) and (len(event_hits[Type]) > maxclass):
            continue

        nhits_list.append(nhits)

        Hits = np.zeros((Event.GetNHTs(), 4))

        for i in range(0, Event.GetNHTs()):
            Hits[i, 0] = Event.GetHTAt(i).GetPosition().X()
            Hits[i, 1] = Event.GetHTAt(i).GetPosition().Y()
            Hits[i, 2] = Event.GetHTAt(i).GetPosition().Z()
            Hits[i, 3] = Event.GetHTAt(i).GetEnergy()

        NEvents += 1

        if Type not in event_types:
            event_hits[Type] = []

        event_hits[Type].append(Hits)

        event_types.append(Type)

        if (maxevents != None) and (NEvents >= maxevents):
            break

        if (maxclass != None) and (len(event_hits[0]) >= maxclass) and (len(event_hits[1]) >= maxclass):
            print("Breaking on max number of events for each class")
            print("0:", len(event_hits[0]))
            print("1:", len(event_hits[1]))
            break


    print("Total number processed = ", i)

    print("Occurrences of different event types:")
    print(collections.Counter(event_types))
    print("Occurances of Nhits:")
    count_nhits = collections.Counter(nhits_list)
    print(count_nhits)
    nhits_array = np.array(nhits_list)

    print("Max Label:", max_label)

    '''
    shuffledTypes = EventTypes.copy()
    shuffledHits = EventHits.copy()

    random.shuffle(shuffledHits)
    random.shuffle(shuffledTypes)

    ceil = math.ceil(len(self.EventHits)*0.75)
    EventTypesTrain = shuffledTypes[:ceil]
    EventTypesTest = shuffledTypes[ceil:]
    EventHitsTrain = shuffledHits[:ceil]
    EventHitsTest = shuffledHits[ceil:]
    '''

    return event_hits

def gen_dataset(event_hits, nevents):
    """This justs takes a random subset of each event type so that we have
    similar number of events of each type. It returns a list of the hits and types
    for the dataset."""

    event_types = np.array(list(event_hits.keys()))

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
        # events_dataset.append(events)
        # types_dataset.append(types)

        dataset_hits = dataset_hits + events
        dataset_types = dataset_types + list(types)
    
    return dataset_hits, dataset_types


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Cosima Sims')

    parser.add_argument('-fn', dest='fn', action='store', help='Filename to parse')
    parser.add_argument('-nevents', dest='nevents', type=int, action='store', help='Number of events')
    parser.add_argument('-minhits', dest='minhits', type=int, action='store', default=1,
                         help='Mininum number of hits for good event.')
    parser.add_argument('-outfn', dest='outfn', action='store', help='Path to store output file.')
    parser.add_argument('-preprocess_only', dest='preprocess_only', action='store_true')
    parser.add_argument('-nevents_dataset', dest='nevents_dataset', type=int, action='store', default=500000)

    args = parser.parse_args()

    # 1. Load the Cosima sim data, do some filtering, and separate into different event types
    event_hits = load_data(args.fn, args.nevents, minhits=args.minhits,
                           maxclass=args.nevents_dataset)
    if args.preprocess_only:
        if args.outfn is not None:
            with open(args.outfn, 'wb') as f:
                pickle.dump(event_hits, f)
        exit()

    #2. Get similar number of each event type and combine to form a dataset
    dataset = gen_dataset(event_hits, args.nevents_dataset)
    print("Size of dataset:", len(dataset[0]))
    if args.outfn is not None:
        with open(args.outfn, 'wb') as f:
            pickle.dump(dataset, f)