from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nsga2_model_interface import ModelInterface


def main():

    models = ModelInterface()

    print("\n========== MODEL CHECK ==========")

    print("LightGBM model loaded:", models.lgbm is not None)
    print("Siamese classifier loaded:",
          models.pair_classifier is not None)

    print("TF-IDF vectorizer loaded:",
          models.vectorizer is not None)

    print("LightGBM threshold:",
          models.lgbm_threshold)

    print("Selected K:", models.selected_k)

    print("\nAll model-loading checks passed.")


if __name__ == "__main__":
    main()
