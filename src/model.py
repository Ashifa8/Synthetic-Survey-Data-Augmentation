"""
src/model.py
DGNN synthesizer training and ML classifier construction.
"""

import os
import pickle
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC


# ------------------------------------------------------------------ #
#  Synthesizer helpers                                                 #
# ------------------------------------------------------------------ #

def train_synthesizers(seed_df, metadata, cfg: dict) -> dict:
    """
    Train CTGAN, TVAE, and CopulaGAN on seed_df.
    Returns dict of {name: fitted_synthesizer}.
    Saves checkpoints to cfg['paths']['checkpoints_dir'].
    """
    epochs    = cfg["synthesizers"]["epochs"]
    ckpt_dir  = cfg["paths"]["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    synthesizers = {}

    print("Training CTGAN...")
    ctgan = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)
    ctgan.fit(seed_df)
    ctgan.save(os.path.join(ckpt_dir, "ctgan.pkl"))
    synthesizers["ctgan"] = ctgan
    print("CTGAN done.\n")

    print("Training TVAE...")
    tvae = TVAESynthesizer(metadata, epochs=epochs)
    tvae.fit(seed_df)
    tvae.save(os.path.join(ckpt_dir, "tvae.pkl"))
    synthesizers["tvae"] = tvae
    print("TVAE done.\n")

    print("Training CopulaGAN...")
    copulagan = CopulaGANSynthesizer(metadata, epochs=epochs, verbose=True)
    copulagan.fit(seed_df)
    copulagan.save(os.path.join(ckpt_dir, "copulagan.pkl"))
    synthesizers["copulagan"] = copulagan
    print("CopulaGAN done.\n")

    return synthesizers


def load_synthesizers(cfg: dict) -> dict:
    """Load pre-trained synthesizers from checkpoints directory."""
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    names = ["ctgan", "tvae", "copulagan"]
    synth_classes = {
        "ctgan":     CTGANSynthesizer,
        "tvae":      TVAESynthesizer,
        "copulagan": CopulaGANSynthesizer,
    }
    synthesizers = {}
    for name in names:
        path = os.path.join(ckpt_dir, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}. Run train.py first.")
        synthesizers[name] = synth_classes[name].load(path)
        print(f"Loaded {name} from {path}")
    return synthesizers


# ------------------------------------------------------------------ #
#  ML Classifier factory                                               #
# ------------------------------------------------------------------ #

def get_classifiers(random_state: int = 42) -> dict:
    """Return the four classifiers used in the paper."""
    return {
        "Boosting":     GradientBoostingClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "SVM-RBF":      SVC(kernel="rbf",    probability=True, random_state=random_state),
        "SVM-Linear":   SVC(kernel="linear", probability=True, random_state=random_state),
    }
