import torch
import torch.nn.functional as F

x = torch.load('savefile/weight/deit_base_patch16_224-b5f2ef4d.pth')

print(x['model'].keys())

all_pos_emb = x['model']['pos_embed']
print(all_pos_emb.shape)

pos_emb = all_pos_emb[:,1:,:]
print(pos_emb.shape)

pos_emb = pos_emb.reshape(-1, 14, 14, 768)
pos_emb = pos_emb.permute(0,3,1,2) # 1, 768, 14, 14,
print(pos_emb.shape)

pos_emb = F.interpolate(pos_emb, size=(24, 24), mode='bilinear') # to 14 -> 24
print(pos_emb.shape)

pos_emb = pos_emb.permute(0,2,3,1) # 1, 768, 32, 32 -> 1, 32, 32, 768
pos_emb = pos_emb.reshape(1, -1, 768)
print(pos_emb.shape)
print(all_pos_emb[:,0,:].reshape(1,1,-1).shape)

pos_emb = torch.cat([all_pos_emb[:,0,:].reshape(1,1,-1), pos_emb], dim=1)
print(pos_emb.shape)

x['model']['pos_embed'] = pos_emb

torch.save(x, 'savefile/weight/deit_base_patch16_224-b5f2ef4d_to384.pth')