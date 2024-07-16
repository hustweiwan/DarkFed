import torch
from synthesizers.pattern_synthesizer import PatternSynthesizer
#这个函数应该是制作单个像素的pattern

class SinglePixelSynthesizer(PatternSynthesizer):
    pattern_tensor = torch.tensor([[1.]])
