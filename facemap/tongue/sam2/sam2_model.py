import os
import torch
import torchvision
import sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
import sys
sys.path.insert(0, "/home/asyeda/segment-anything-2")
from sam2.build_sam import build_sam2_video_predictor

class SAM2Model:
    def __init__(self, video_path):
        self.set_device()
        self.load_model()
        self.video_path = video_path
        self.video_dir = os.path.join(os.path.dirname(video_path), "jpgs")
        self.init_state()

    def init_state(self):
        # check if [".jpg", ".jpeg", ".JPG", ".JPEG"] files are present in the directory. if not, convert the video to frames
        if not any([i.endswith((".jpg", ".jpeg", ".JPG", ".JPEG")) for i in os.listdir(self.video_dir)]):
            os.system(f"ffmpeg -i {self.video_path} -q:v 2 -start_number 0 {self.video_dir}/%05d.jpg")
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        print("Inference state initialized")

    def set_device(self):
        # select the device for computation
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")

        if self.device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

    def load_model(self):
        # load the model
        #!mkdir -p ../checkpoints/
        #!wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt       
        # get the model configuration file
        sam2_checkpoint = "/home/asyeda/Facemap/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)

    def track_objects(self):
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=0):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().astype(int).squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    def reset(self):
        self.predictor.reset_state(self.inference_state)
