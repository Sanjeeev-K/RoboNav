import torch

for x in range(150,3250,50):
    y = torch.load("stage_1_"+str(x)+"_critic.pth", map_location = torch.device('cpu'))
    torch.save(y,"stage_4_"+str(x)+"_critic.pth")
    y = torch.load("stage_1_"+str(x)+"_actor.pth", map_location = torch.device('cpu'))
    torch.save(y,"stage_4_"+str(x)+"_actor.pth")
