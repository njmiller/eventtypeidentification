import torch

from models.pointnet import PointNet

fn = "test_torch_model_params_May14.pth"
model.load_state_dict(torch.load(fn))
model.eval()

torch_input = torch.randn(1, 1, 110, 110, 48)
# onnx_program = torch.onnx.dynamo_export(model, torch_input)
# input_names = [ "voxels" ]
# output_names = [ "output" ]
# torch.onnx.export(model, torch_input, "test_model.onnx", verbose=True,
                #   input_names=input_names, output_names=output_names,
                #   export_params=True)

# exported_mod = torch.export.export(model, (torch_input,))
# print(type(exported_mod))
# print(exported_mod.module()(torch.randn(10, 1, 110, 110, 48)))

# print("TEST:", model(torch_input))
# zzz

model_scripted = torch.jit.script(model)
model_scripted.save("model_scripted.pt")