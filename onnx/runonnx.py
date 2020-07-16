# ...continuing from above
import onnxruntime as ort
import numpy as np
import ipdb
ort_session = ort.InferenceSession('msg3dv214.onnx')

outputs = ort_session.run(None, {'actual_input_1': np.random.randn(1, 3, 10, 14, 1).astype(np.float32)})

print(outputs[0])