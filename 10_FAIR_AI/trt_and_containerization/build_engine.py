import engine as eng
import argparse
from onnx import ModelProto
import tensorrt as trt 
 
name = 'cifar_net_10_epochs' 
engine_name = '{}_4_trial.plan'.format(name)
onnx_path = '/home/nravi/ai-science-training-series/10_FAIR_AI/trt_and_containerization/saved_models/{}.onnx'.format(name)
batch_size = 4

model = ModelProto()
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

shape = [batch_size , d0, d1, d2]
print(shape)
engine = eng.build_engine(onnx_path, shape= shape)
eng.save_engine(engine, engine_name)
print(shape)
