"""Code to run inference on a Cosima file using a PointNet model."""
import argparse

import torch

import ROOT as M

from models.pointnet import PointNet
    
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

def main():

    fn = '/data/slag2/njmille2/AMEGOXData0p5/AMEGOX_1MeV_50MeV_flat.p1.inc10.id1.sim.gz'
    fn2 = '/data/slag2/njmille2/AMEGOXData0p5/AMEGOX_1MeV_50MeV_flat.p1.inc10.id1.tra.gz'
    fn_out = './EventTypes.txt'

    # f = open(fn_out, "w")

    G = M.MGlobal()
    G.Initialize()

    geometry_name = "~/ComPair/Geometry/AMEGO_Midex/AmegoXBase.geo.setup"
    
    # Load geometry:
    Geometry = M.MDGeometryQuest()
    if Geometry.ScanSetupFile(M.MString(geometry_name)) == True:
        print("Geometry " + geometry_name + " loaded!")
    else:
        print("Unable to load geometry " + geometry_name + " - Aborting!")
        quit()
    
    Reader = M.MFileEventsSim(Geometry)
    if Reader.Open(M.MString(fn)) == False:
        print("Unable to open file " + fn + ". Aborting!")
        quit()

    reader_tra = M.MFileEventsTra()
    if reader_tra.Open(M.MString(fn2)) == False:
        print("Unable to open file " + fn2 + ". Aborting!")
        quit()


    model = PointNet()
    model.load_state_dict(torch.load("/data/slag2/njmille2/AMEGOXData0p5/test_torch_model_params_20250714_pn_inf.pth"))
    model.eval()
            
    nbins = 50
    corr_both = torch.zeros([nbins, 2, 2], dtype=torch.int)
    corr_pn = torch.zeros([nbins, 2], dtype=torch.int)
    corr_revan = torch.zeros([nbins, 2], dtype=torch.int)
    corr_revan_mixed = torch.zeros(nbins, dtype=torch.int)
    
    tot_both = torch.zeros([nbins, 2], dtype=torch.int)
    tot_pn = torch.zeros([nbins, 2], dtype=torch.int)
    tot_revan = torch.zeros([nbins, 2], dtype=torch.int)
    tot_revan_mixed = torch.zeros(nbins, dtype=torch.int)
    tot_neither = torch.zeros(nbins, dtype=torch.int)
    tot_neither_type = torch.zeros([nbins, 3], dtype=torch.int)
    
    i = 0
    id = 0
    id_tra = -1
    while True:
        Event = Reader.GetNextEvent()
        M.SetOwnership(Event, True)
        i += 1

        if not Event:
            break
        
        id = Event.GetID()

        energy = get_energy(Event)
        idx_bin = int(energy / 1000) # Energy is from 1 to 50 MeV. Energy value should be in keV
        
        if id_tra < id:
            event_tra = reader_tra.GetNextEvent()
            M.SetOwnership(event_tra, True)
            
            id_tra = event_tra.GetId()
        
            if not event_tra:
                break

            if id_tra < id:
                raise ValueError("Read new TRA ID and still less than Sim ID")
        
 
        if i % 4000 == 0:
            print(f"{i}: STUFF")
            # print(f"{i}: tot: {tot}")
            # print(f"good_pn: {good_pn}")
            # print(f"good_revan: {good_revan}")
            # print(f"Acc PN: {good_pn[2]/tot[2]*100}, {good_pn[3]/tot[3]*100}")
            # print(f"Acc Revan: {good_revan[1]/tot[1]*100}, {good_revan[3]/tot[3]*100}")
            # print()
        
        is_good_revan = id == id_tra
        
        is_good_pn, reason = is_good_event(Event)
        
        # Get the event type for the sim event only if
        event_type = get_event_type(Event)

        # Pointnet is only good for event types of 0 and 1
        if (event_type == -1) | (event_type == 2):
            if is_good_pn:
                is_good_pn = False
                reason = 2
     
        # Check whether Revan identifies this sim ID
        if is_good_revan:
            good_revan_tmp, etr, is_mixed = get_acc_revan(event_tra, event_type)
            is_good_revan = is_good_revan & (good_revan_tmp >= 0)

        if is_good_pn:
            good_pn_tmp = get_acc_pn(Event, model, event_type)


        if (not is_good_revan) & (not is_good_pn):
            tot_neither[idx_bin] += 1
            tot_neither_type[idx_bin, reason] += 1
        elif is_good_revan & (not is_good_pn):
            if is_mixed:
                tot_revan_mixed[idx_bin] += 1
                corr_revan_mixed[idx_bin] += good_revan_tmp
            else:
                tot_revan[idx_bin, etr] += 1
                corr_revan[idx_bin, etr] += good_revan_tmp
        elif is_good_pn & (not is_good_revan):
            tot_pn[idx_bin, event_type] += 1
            corr_pn[idx_bin, event_type] += good_pn_tmp
        else:
            tot_both[idx_bin, event_type] += 1
            corr_both[idx_bin, event_type, 0] += good_pn_tmp
            corr_both[idx_bin, event_type, 1] += good_revan_tmp


    # print(f"tot: {tot}")
    # print(f"good_pn: {good_pn}")
    # print(f"good_revan: {good_revan}")
    # print(f"Acc PN: {good_pn[3]/tot[3]*100}, {good_pn[4]/tot[4]*100}")
    # print(f"Acc Revan: {good_revan[1]/tot[1]*100}, {good_revan[2]/tot[2]*100}, {good_revan[4]/tot[4]*100}")
    print()

    torch.save({
        "tot_neither": tot_neither,
        "tot_neither_type": tot_neither_type,
        "tot_revan": tot_revan,
        "tot_revan_mixed": tot_revan_mixed,
        "tot_pn": tot_pn,
        "tot_both": tot_both,
        "corr_revan": corr_revan,
        "corr_revan_mixed": corr_revan_mixed,
        "corr_pn": corr_pn,
        "corr_both": corr_both
    }, "pn_revan_comparison.pt")

def get_energy(event):
    
    energy = event.GetIAAt(0).GetSecondaryEnergy()

    return energy

def get_acc_pn(event, model, event_type):
        '''Returns 1 is model predicts correct event and 0 if it does not.'''
        data_input = process_event(event)

        logits, _ = model(data_input)

        if logits >= 0.0 and event_type == 1:
            return 1
        elif logits < 0.0 and event_type == 0:
            return 1
        
        return 0

def get_acc_revan(event_tra, event_type):
        if event_tra.GetType() == M.MPhysicalEvent.c_Compton:
            event_type_revan = 0
        elif event_tra.GetType() == M.MPhysicalEvent.c_Pair:
            event_type_revan = 1
        else:
            return -1, 0, False

        is_mixed = False 
        if event_type == 2:
            event_type = 0
            is_mixed = True
        
        if event_type_revan == event_type:
            return 1, event_type_revan, is_mixed
        else:
            return 0, event_type_revan, is_mixed
            

def is_good_event(event):
    
    nhits = event.GetNHTs()
    
    minhits = 2
    if nhits < minhits:
        return False, 0

    # xpos = event.GetHTAt(0).GetPosition().X()
    # ypos = event.GetHTAt(0).GetPosition().Y()
    # zpos = event.GetHTAt(0).GetPosition().Z()
    
    xpos = event.GetIAAt(1).GetPosition().X()
    ypos = event.GetIAAt(1).GetPosition().Y()
    zpos = event.GetIAAt(1).GetPosition().Z()

    # Checking for first interaction in the tracker volume
    if abs(xpos) < 43.74 and abs(ypos) < 43.74 and zpos >= -36.432 and zpos <= 22.068:
        return True, -1

    detectortype = 1 
    if (detectortype is not None) and event.GetIAAt(1).GetDetectorType() != detectortype:
        # event.GetIAAt(1).GetPosition #check to see whether it is in tracker volume
        return False, 1

    return True, -1

def get_event_type(event):
        
    if event.GetNIAs() > 0:
        if event.GetIAAt(1).GetProcess() == M.MString("COMP"):
            if is_mixed(event):
                return 2
            else:
                return 0
        elif event.GetIAAt(1).GetProcess() == M.MString("PAIR"):
            return 1
    else:
        return -1

def process_event(event):

    nhits = event.GetNHTs()

    data = torch.zeros([1, 4, nhits])

    for i in range(nhits):
        data[0, 0, i] = event.GetHTAt(i).GetPosition().X()
        data[0, 1, i] = event.GetHTAt(i).GetPosition().Y()
        data[0, 2, i] = event.GetHTAt(i).GetPosition().Z()
        data[0, 3, i] = event.GetHTAt(i).GetEnergy()

    return data

def is_mixed(event):
    #NOTE: Only call if initial event type test says that it is Compton 
    nias = event.GetNIAs()
    for i in range(2, nias):
        if event.GetIAAt(i).GetProcess() == M.MString("PAIR") and event.GetIAAt(i).GetDetectorType() == 1:
            return True
        
    return False

if __name__ == '__main__':
    main()