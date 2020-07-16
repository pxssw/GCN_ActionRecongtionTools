import torch
from model.msg3d import Model
import sys
import ipdb
sys.path.append('./model')
import numpy as np
import random
seed = 25
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
dummy_input = torch.zeros(1, 3, 300, 14, 1, device='cuda')
model = Model(
    num_class=11,
    num_point=14,
    num_person=1,
    num_gcn_scales=8,
    num_g3d_scales=6,
    graph='graph.ntu_rgb_d.AdjMatrixGraph'
).cuda()
model.load_state_dict(torch.load('../MS-G3Dv2.14/train_ntu_joint2D14/weights/weights-110-56540.pt', map_location='cpu'))
# model.load_state_dict(torch.load('../MS-G3Dv2.14/train_ntu_joint2D14_300/weights/weights-93-47802.pt', map_location='cpu'))
# model.load_state_dict(torch.load('./weights-93-47802.pt'))weights-110-56540.pt

output = model(dummy_input)
# ipdb.set_trace()

# model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

# torch.onnx.export(model, dummy_input, "weights-102-59976.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(model, dummy_input, "msg3dv214_300.onnx", verbose=True, input_names=input_names, output_names=output_names)

