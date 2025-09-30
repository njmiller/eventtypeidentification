import torch
import torch.onnx

# import onnx
import onnxruntime as ort

fn = '../pointnet_traced.pt'
fn_out = 'pointnet_trained.onnx'

model = torch.jit.load(fn)
model.eval()

dummy_point_cloud = torch.randn([2, 4, 10])
dummy_mask = torch.ones([2, 10])

# tmp, tmp2 = model(torch_input, mask)
# print("SSS:", tmp, tmp.shape, tmp2.shape)
# zzz

dummy_input = (dummy_point_cloud, dummy_mask)

torch.onnx.export(
    model,
    dummy_input,
    fn_out,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names = ['point_cloud', 'mask'],
    output_names = ['logits', 'trans_feat'],
    dynamic_axes={
        'point_cloud': {0: 'batch_size', 2: 'num_points'},
        'mask': {0: 'batch_size', 1: 'num_points'},
        'logits': {0: 'batch_size'},
        'trans_feat': {0: 'batch_size'}
    }
)

# Test
with torch.no_grad():
    tmp, tmp2 = model(*dummy_input)

ort_session = ort.InferenceSession(fn_out)

onnx_inputs = {
    'point_cloud': dummy_input[0].numpy(),
    'mask': dummy_input[1].numpy()
}

onnx_outputs = ort_session.run(None, onnx_inputs)

print(tmp)
print(onnx_outputs[0])