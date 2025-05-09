from concepts import concepts

# from modeling_chexagent import CheXagentVisionEmbeddings
# from configuration_chexagent import CheXagentVisionConfig

import io
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from PIL import Image
from torchvision import transforms

# from mistralai.client import MistralClient

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
import torch
from dotenv import load_dotenv

load_dotenv()

import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# client = MistralClient(api_key=MISTRAL_API_KEY)

transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

# # Vision Model
# config = CheXagentVisionConfig(
#     hidden_size=1024,
#     intermediate_size=6144,
#     num_hidden_layers=39,
#     num_attention_heads=16,
#     image_size=224,
#     patch_size=14,
#     hidden_act="gelu",
#     layer_norm_eps=1e-6,
#     attention_dropout=0.0,
#     initializer_range=1e-10,
#     qkv_bias=True,
# )

# vision_model = CheXagentVisionEmbeddings(config)

# step 1: Setup constant
vlm_id = "StanfordAIMI/CheXagent-2-3b"
dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# step 2: Load Processor and Model
tokenizer = AutoTokenizer.from_pretrained(vlm_id, trust_remote_code=True)
vision_model = AutoModelForCausalLM.from_pretrained(
    vlm_id, device_map={"": "cuda"}, torch_dtype=dtype, trust_remote_code=True
)

llm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm.to(device)


def get_image_concept_similarity_vector(uploaded_file):
    # embeddings_batch_response = client.embeddings(
    #     model="mistral-embed",
    #     input=concepts,
    # )
    embeddings = llm.encode(concepts, convert_to_tensor=True, device="cuda")

    I = (
        transform(Image.open(uploaded_file).convert("RGB"))
        .unsqueeze(0)
        .to("cuda", dtype=torch.float32)
    )

    with torch.no_grad():
        vision_features = vision_model.model.visual(I)

    V = vision_features.squeeze(0).to(dtype=torch.float32)
    projector = torch.nn.Linear(1024, 384).to(device)
    V_projected = projector(V.T)
    e = []
    for concept in tqdm(embeddings):
        concept = concept.to(device="cuda", dtype=torch.float32)
        s = torch.nn.functional.cosine_similarity(
            V_projected, concept.unsqueeze(0), dim=1
        )
        e.append(max(s))

    return e
