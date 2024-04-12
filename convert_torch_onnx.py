import torch

import torch_models as tm

nclasses = 2
model = tm.VoxNet(num_classes=nclasses, input_shape=(110, 110, 48))

fn = "torch_test_model2.pth"
model.load_state_dict(torch.load(fn))
model.eval()

torch_input = torch.randn(1, 1, 110, 110, 48)
# onnx_program = torch.onnx.dynamo_export(model, torch_input)
input_names = [ "voxels" ]
output_names = [ "output" ]
torch.onnx.export(model, torch_input, "test_model.onnx", verbose=True,
                  input_names=input_names, output_names=output_names,
                  export_params=True)