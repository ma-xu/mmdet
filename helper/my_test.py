import torch

checkpoint = torch.load('SOLO_R50_1x.pth', map_location='cpu')
checkclass = checkpoint['meta']['CLASSES']
print(checkclass)