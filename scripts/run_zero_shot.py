import torch
from transformer_maskgit import CTViT
import pandas as pd
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(os.path.dirname(__file__), '../models/CT-CLIP_v2.pt')
# data_folder = os.path.join(os.path.dirname(__file__), '../Dataset_trial/data_volumes/dataset/train')
data_folder = os.path.join(os.path.dirname(__file__), '../Dataset_trial/npz_folder')
reports_file = os.path.join(os.path.dirname(__file__), '../Dataset_trial/validation_reports.csv')
labels = os.path.join(os.path.dirname(__file__), '../Dataset_trial/dataset_multi_abnormality_labels_valid_predicted_labels.csv')
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))

# data = pd.read_csv(reports_file)

# print(data.head())

image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)

clip.load(model_path)

inference = CTClipInference(
    clip,
    data_folder = data_folder,
    reports_file= reports_file,
    labels = labels,
    batch_size = 1,
    results_folder="inference_zeroshot/",
    num_train_steps = 10,
    save_results_every = 5,
    save_model_every = 5
)

inference.infer()