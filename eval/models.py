import torch
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.inference import loss_of_one_batch
from dust3r.utils.device import to_cpu, collate_with_cat
from mast3r.model import AsymmetricMASt3R


inf = float("inf")

class RES():
    def __init__(self, output):
        self.output = output

    def get_depth(self):
        return self.output['pred1']['pts3d'][0, :, :, 2].detach().cpu().numpy()
    
    def get_conf(self):
        return (self.output['pred1']['conf'].squeeze(0).detach().cpu().numpy(), self.output['pred2']['conf'].squeeze(0).detach().cpu().numpy())
    
    def get_clip(self):
        return (self.output['pred1']['clip'].squeeze(0).detach().cpu().numpy(), self.output['pred2']['clip'].squeeze(0).detach().cpu().numpy())
    
    def get_dino(self):
        return (self.output['pred1']['dino'].squeeze(0).detach().cpu().numpy(), self.output['pred2']['dino'].squeeze(0).detach().cpu().numpy())

class dust3r():
    def __init__(self, model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", device = "cuda"):
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        self.devide = device

    @torch.no_grad()
    def predict(self, images):
        # input list of two images
        res = loss_of_one_batch(collate_with_cat([tuple(images)]), self.model, None, self.device)
        return RES(to_cpu(res))

class mast3r():
    def __init__(self, model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", device = "cuda"):
        self.model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        self.devide = device

    @torch.no_grad()
    def predict(self, images):
        # input list of two images
        res = loss_of_one_batch(collate_with_cat([tuple(images)]), self.model, None, self.device)
        return RES(to_cpu(res))
 
class Sab3r():
    def __init__(self, model_config, model_path, device = "cuda"):
        self.device = device
        
        def load_model(model, ckpt_path, device):
            
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if ckpt_path.endswith('.pth'):
                model.load_state_dict(ckpt['model'], strict=False)
            elif ckpt_path.endswith('.pt'):
                model.load_state_dict(ckpt['module'])
            else:
                raise ValueError(f"Unknown checkpoint format: {ckpt_path}")
            
            model = model.to(device)

        def enable_mast3r(args):
            args = args.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
            if 'landscape_only' not in args:
                args = args[:-1] + ', landscape_only=False)'
            else:
                args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
            return args
        
        model_config = enable_mast3r(model_config)
        
        self.model = eval(model_config)
        
        load_model(self.model, model_path, device)

    @torch.no_grad()
    def predict(self, images):
        # input list of two images
        res = loss_of_one_batch(collate_with_cat([tuple(images)]), self.model, None, self.device)
        return RES(to_cpu(res))