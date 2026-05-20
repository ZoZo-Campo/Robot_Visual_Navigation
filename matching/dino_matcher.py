import os
from typing import cast

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOMatcher:

    def __init__(
        self,
        model_name="facebook/dinov2-small",
        device=None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.processor = AutoImageProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        )

        self.model.eval()
        self.model.to(self.device)

    def extract_features(
        self,
        image_path,
    ):
        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        features = outputs.last_hidden_state[:, 0, :]

        features = (
            features.squeeze()
            .cpu()
            .numpy()
        )

        norm = np.linalg.norm(features)

        if norm > 0:
            features = features / norm

        return features

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

    def compare_images(
        self,
        query_image_path,
        database_image_path,
    ):
        query_features = self.extract_features(
            query_image_path
        )

        database_features = self.extract_features(
            database_image_path
        )

        score = self.cosine_similarity(
            query_features,
            database_features
        )

        return {
            "score": score
        }

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

            comparison = self.compare_images(
                query_image_path,
                image_path
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