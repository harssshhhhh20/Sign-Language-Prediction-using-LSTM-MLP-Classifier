"""slr - landmark-based sign language recognition research toolkit.

Package layout
--------------
schema      : MediaPipe landmark layout, feature-group definitions
manifest    : dataset manifest (signer / session / sequence provenance)
features    : feature-group selection, normalisation, temporal pooling
splits      : evaluation protocols (random .. signer-independent)
models      : model registry (classical + neural)
evaluate    : metric computation
experiments : experiment drivers
pipeline    : dual-pathway runtime (motion gate + live demo)
report      : table / figure generation
"""

__version__ = "0.2.0"
