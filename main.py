from data import load_data
from cbm_inference_app import get_image_concept_similarity_vector

from multi_agent_rag import generate_report as generate_report_multi

from single_agent_rag import generate_report as generate_report_single
from gpt4 import generate_report as generate_report_gpt4

from config import num_classes, num_concepts, class_mapping
from concepts import concepts

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm

import re

# import argparse

import warnings
import transformers

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data.iloc[idx]
        r = get_image_concept_similarity_vector(image_path)
        r_tensor = torch.stack(r).float()
        return r_tensor, torch.tensor(label, dtype=torch.long)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# parser = argparse.ArgumentParser(description="Select an agent type.")

# parser.add_argument(
#     "--agent",
#     choices=["multi", "single", "gpt4"],
#     default="multi",
#     help="Choose the agent type: 'multi', 'single', or 'gpt4'. Default is 'multi'.",
# )

# args = parser.parse_args()

cardio_img = "cardio.jpg"
thorax_img = "thorax.jpg"

# Get the image concept similarity vector
r = get_image_concept_similarity_vector(cardio_img)
r_tensor = torch.stack(r).float()
r_tensor = r_tensor.to("cuda")

# # Load dataset
# batch_size = 8
# train_data_df = load_data("sampled_data.csv")
# train_dataset = ImageDataset(train_data_df)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Training Loop
# epochs = 3
# lr = 1e-3
# model = LinearClassifier(num_concepts, num_classes)
# model.to("cuda")
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# for epoch in tqdm(range(epochs)):
#     model.train()
#     total_loss = 0

#     for r_vecs, labels in train_loader:
#         r_vecs = r_vecs.to("cuda")
#         labels = labels.to("cuda")

#         optimizer.zero_grad()

#         outputs = model(r_vecs)
#         loss = criterion(outputs, labels)
#         loss.backward()

#         optimizer.step()
#         total_loss += loss.item()
#     tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
# torch.save(model.state_dict(), "linear_classifier.pth")

# Predict the class of the image
model = LinearClassifier(num_concepts, num_classes)
model.load_state_dict(torch.load("linear_classifier.pth"))
model.to("cuda")
model.eval()
W_F = model.fc.weight.detach()
W_F = W_F.to("cuda")
predicted_class = torch.argmax(
    torch.nn.functional.softmax(torch.matmul(r_tensor, W_F.T) + model.fc.bias, dim=0)
).item()
print(predicted_class)

# Concept Contribution
contribution = W_F * r_tensor.unsqueeze(0) + model.fc.bias.unsqueeze(1)
contribution = contribution.detach().cpu().numpy()
conc, cont = shuffle(concepts, contribution[predicted_class])
paired_lists = list(zip(conc, cont))
sorted_paired_lists = sorted(paired_lists, key=lambda x: abs(x[1]), reverse=True)
sorted_concepts, sorted_contributions = zip(*sorted_paired_lists)
sorted_concepts = list(sorted_concepts)
sorted_contributions = list(sorted_contributions)

# Generate the report
report = generate_report_gpt4(class_mapping, 1, sorted_concepts, sorted_contributions)
# if args.agent == "multi":
#     report, logs = generate_report_multi(
#         class_mapping, predicted_class, sorted_concepts, sorted_contributions
#     )
# elif args.agent == "single":
#     report = generate_report_single(
#         class_mapping, predicted_class, sorted_concepts, sorted_contributions
#     )
# elif args.agent == "gpt4":
#     report = generate_report_gpt4(
#         class_mapping, predicted_class, sorted_concepts, sorted_contributions
#     )
print()
print("< 생성 결과 >")
print(report)
