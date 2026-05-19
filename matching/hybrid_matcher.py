class HybridMatcher:

    def __init__(
        self,
        orb_weight=0.4,
        cnn_weight=0.6,
    ):

        total = orb_weight + cnn_weight

        self.orb_weight = (
            orb_weight / total
        )

        self.cnn_weight = (
            cnn_weight / total
        )

    # =====================================================
    # SCORE FUSION
    # =====================================================

    def compute_hybrid_score(
        self,
        orb_score,
        cnn_score,
    ):

        final_score = (
            self.orb_weight * orb_score
            +
            self.cnn_weight * cnn_score
        )

        return final_score

    # =====================================================
    # RESULT FUSION
    # =====================================================

    def fuse_results(
        self,
        orb_results,
        cnn_results,
        top_k=5,
    ):

        orb_dict = {
            item["filename"]: item
            for item in orb_results
        }

        cnn_dict = {
            item["filename"]: item
            for item in cnn_results
        }

        common_files = (
            set(orb_dict.keys())
            &
            set(cnn_dict.keys())
        )

        fused_results = []

        for filename in common_files:

            orb_score = (
                orb_dict[filename]["score"]
            )

            cnn_score = (
                cnn_dict[filename]["score"]
            )

            final_score = (
                self.compute_hybrid_score(
                    orb_score,
                    cnn_score,
                )
            )

            result = {
                "filename": filename,

                "score": final_score,

                "orb_score": orb_score,

                "cnn_score": cnn_score,

                "good_matches": orb_dict[
                    filename
                ].get(
                    "good_matches",
                    0
                ),
            }

            fused_results.append(result)

        fused_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return fused_results[:top_k]