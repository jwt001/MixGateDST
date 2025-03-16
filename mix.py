import mixgate
from config import get_parse_args
import torch

args = get_parse_args()
#Create MixGate
model = mixgate.TopModel(
    args,
    dg_ckpt_aig='/home/jwt/MixGate/ckpt/model_func_aig.pth',
    dg_ckpt_xag='/home/jwt/MixGate/ckpt/model_func_xag.pth',
    dg_ckpt_xmg='/home/jwt/MixGate/ckpt/model_func_xmg.pth',
    dg_ckpt_mig='/home/jwt/MixGate/ckpt/model_func_mig.pth' )    

model.load('/home/jwt/MixGate_aig/model_30.pth')      # Load pretrained model
parser = mixgate.MixParser()   # Create AigParser
graph = parser.read_bench('/home/jwt/downstream_task/examples/aig_folder/adder_8_44.aig', '/home/jwt/also/build/bin/also') # Parse AIG into Graph
hs, hf = model(graph)       # Model inference 
# hs: structural embeddings, hf: functional embeddings
# hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)
print("hs: ", hs)
print("hf: ", hf)
