"""Code to run inference on a Cosima file using a PointNet model."""
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import ROOT as M

from models.pointnet import PointNet
    
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

def main(fn, fn_out, geometry_name, model_weights):

    f = open(fn_out, "w")

    G = M.MGlobal()
    G.Initialize()

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

    model = PointNet()
    model.load_state_dict(torch.load(model_weights))
    # model = torch.jit.load(model_traced)
    model.eval()


    i = 0
    while True:
        Event = Reader.GetNextEvent()
        M.SetOwnership(Event, True)
        i += 1

        if not Event:
            break
        
        id = Event.GetID()

        # energy = get_energy(Event)
        # idx_bin = int(energy / 1000) # Energy is from 1 to 50 MeV. Energy value should be in keV
        
        if i % 4000 == 0:
            print(f"{i}: STUFF")
        
        is_good, reason = is_good_event(Event)
        
        # Get the event type for the sim event only if
        # event_type = get_event_type(Event)

        if is_good:
            data_input = process_event(Event)

            logits, _ = model(data_input)

            prob = torch.sigmoid(logits)

            if logits > 0:
                et_out = 'PA'
                tp_out = prob.item()
            else:
                et_out = 'CO'
                tp_out = 1-prob.item()

            print(f"SE\nID {id}\nET {et_out}\nTP {tp_out}", file=f)

    f.close()

    print("DONE")

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
    fn = '/data/slag2/njmille2/AMEGOXData0p5/AMEGOX_1MeV_50MeV_flat.p1.inc10.id1.sim.gz'
    fn_out = 'test2.etp'
    geometry_name = "~/ComPair/Geometry/AMEGO_Midex/AmegoXBase.geo.setup"
    model_weights = "/data/slag2/njmille2/AMEGOXData0p5/test_torch_model_params_20250714_pn_inf.pth"

    main(fn, fn_out, geometry_name, model_weights)