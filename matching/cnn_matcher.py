import os

import torch
import numpy as np

from typing import cast

from torchvision import models
from torchvision import transforms

from PIL import Image


class CNNMatcher:

    def __init__(
        self,
        image_size=224,
        device=None,
    ):

        self.image_size = image_size

        if device is None:

            if torch.backends.mps.is_available():
                self.device = "mps"

            elif torch.cuda.is_available():
                self.device = "cuda"

            else:
                self.device = "cpu"

        else:
            self.device = device

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove final classification layer
        self.model = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # =====================================================
    # IMAGE LOADING
    # =====================================================

    def load_image(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = cast(
            torch.Tensor,
            self.transform(image)
        )

        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    def extract_features(self, image_path):

        image_tensor = self.load_image(
            image_path
        )

        with torch.no_grad():

            features = self.model(
                image_tensor
            )

        features = (
            features.squeeze()
            .cpu()
            .numpy()
        )

        norm = np.linalg.norm(
            features
        )

        if norm > 0:
            features = features / norm

        return features

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    def cosine_similarity(
        self,
        vector1,
        vector2,
    ):

        similarity = np.dot(
            vector1,
            vector2
        )

        similarity = float(similarity)

        similarity = max(
            0.0,
            min(1.0, similarity)
        )

        return similarity * 100

    # =====================================================
    # IMAGE COMPARISON
    # =====================================================

    def compare_images(
        self,
        query_image_path,
        database_image_path,
    ):

        query_features = (
            self.extract_features(
                query_image_path
            )
        )

        database_features = (
            self.extract_features(
                database_image_path
            )
        )

        similarity = (
            self.cosine_similarity(
                query_features,
                database_features
            )
        )

        return {
            "score": similarity
        }

    # =====================================================
    # DATABASE SEARCH
    # =====================================================

    def search(
        self,
        query_image_path,
        database_dir,
        top_k=5,
        progress_callback=None,
    ):

        image_files = sorted(
            file for file in os.listdir(database_dir)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        )

        results = []

        total = len(image_files)

        for i, filename in enumerate(image_files):

            image_path = os.path.join(
                database_dir,
                filename
            )

            comparison = (
                self.compare_images(
                    query_image_path,
                    image_path
                )
            )

            result = {
                "filename": filename,
                "score": comparison["score"],
            }

            results.append(result)

            if progress_callback is not None:

                progress_callback(
                    current=i + 1,
                    total=total,
                    result=result,
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]