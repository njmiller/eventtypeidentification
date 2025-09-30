import torch

from models.pointnet import PointNet

model = PointNet()
model.load_state_dict(torch.load("/data/slag2/njmille2/AMEGOXData0p5/test_torch_model_params_20250714_pn_inf.pth", weights_only=True))
model.eval()

example1 = torch.randn([1, 4, 10])
example2 = torch.randn([1, 4, 20])
example3 = torch.randn([1, 4, 30])

mask1 = torch.ones([1, 10])

tmp = torch.ones([1, 4, 15])

tmp[0, :, 5] = torch.tensor([1.1, 2.2, 3.3, 4.4])

out = model(tmp)

# print(out)
# zzz

example_inputs = (example1, example2, example3)

model_traced = torch.jit.trace(model, example_inputs=(example1, mask1))
model_traced.save("pointnet_traced.pt")

model_scripted = torch.jit.script(model)
model_scripted.save("pointnet_scripted.pt")


# onnx_model = torch.onnx.export(model,
                            #    (example1, mask1),
                            #    "pointnet_onnx.onnx",
                            #    input_names=["input", "mask"],
                            #    output=["output", "trans_feat"],
                            #    dynamo=True)