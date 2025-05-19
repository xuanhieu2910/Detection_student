import torch
import torchreid


def convertWeight():
  model = torchreid.models.osnet_x0_25(num_classes=1000)
  state_dict = torch.load('C:\\Users\\hieux\\Desktop\\Personal\\Master\\PROJECT\\tracker\\deep_sort_real_time\\deep_sort_real_time\\embedder\\weights\\osnet_x0_25.pth', map_location='cpu')
  model.load_state_dict(state_dict)

  model.eval()
  torch.save(model, 'C:\\Users\\hieux\\Desktop\\Personal\\Master\\PROJECT\\tracker\\deep_sort_real_time\\deep_sort_real_time\\embedder\\weights\\osnet_x0_25.pt')

