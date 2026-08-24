import numpy as np
from typing import NamedTuple
import numpy as np

class BrainLabels(NamedTuple):
    LEFT_CEREBRAL_WHITE_MATTER: np.uint8 = np.uint8(2)
    LEFT_CEREBRAL_CORTEX: np.uint8 = np.uint8(3)
    LEFT_LATERAL_VENTRICLE: np.uint8 = np.uint8(4)
    LEFT_INFERIOR_LATERAL_VENTRICLE: np.uint8 = np.uint8(5)
    LEFT_CEREBELLUM_WHITE_MATTER: np.uint8 = np.uint8(7)
    LEFT_CEREBELLUM_CORTEX: np.uint8 = np.uint8(8)
    LEFT_THALAMUS: np.uint8 = np.uint8(10)
    LEFT_CAUDATE: np.uint8 = np.uint8(11)
    LEFT_PUTAMEN: np.uint8 = np.uint8(12)
    LEFT_PALLIDUM: np.uint8 = np.uint8(13)
    THIRD_VENTRICLE: np.uint8 = np.uint8(14)
    FOURTH_VENTRICLE: np.uint8 = np.uint8(15)
    BRAIN_STEM: np.uint8 = np.uint8(16)
    LEFT_HIPPOCAMPUS: np.uint8 = np.uint8(17)
    LEFT_AMYGDALA: np.uint8 = np.uint8(18)
    CSF: np.uint8 = np.uint8(24)
    LEFT_ACCUMBENS_AREA: np.uint8 = np.uint8(26)
    LEFT_VENTRAL_DC: np.uint8 = np.uint8(28)
    LEFT_CHOROID_PLEXUS: np.uint8 = np.uint8(31)
    RIGHT_CEREBRAL_WHITE_MATTER: np.uint8 = np.uint8(41)
    RIGHT_CEREBRAL_CORTEX: np.uint8 = np.uint8(42)
    RIGHT_LATERAL_VENTRICLE: np.uint8 = np.uint8(43)
    RIGHT_INFERIOR_LATERAL_VENTRICLE: np.uint8 = np.uint8(44)
    RIGHT_CEREBELLUM_WHITE_MATTER: np.uint8 = np.uint8(46)
    RIGHT_CEREBELLUM_CORTEX: np.uint8 = np.uint8(47)
    RIGHT_THALAMUS: np.uint8 = np.uint8(49)
    RIGHT_CAUDATE: np.uint8 = np.uint8(50)
    RIGHT_PUTAMEN: np.uint8 = np.uint8(51)
    RIGHT_PALLIDUM: np.uint8 = np.uint8(52)
    RIGHT_HIPPOCAMPUS: np.uint8 = np.uint8(53)
    RIGHT_AMYGDALA: np.uint8 = np.uint8(54)
    RIGHT_ACCUMBENS_AREA: np.uint8 = np.uint8(58)
    RIGHT_VENTRAL_DC: np.uint8 = np.uint8(60)
    RIGHT_CHOROID_PLEXUS: np.uint8 = np.uint8(63)
    FALX: np.uint8 = np.uint8(70)
    TENTORIUM: np.uint8 = np.uint8(71)
    UNCLASSIFIED: np.uint8 = np.uint8(72)
    SPINAL_BUFFER: np.uint8 = np.uint8(73)
    WM_HYPOINTENSITIES: np.uint8 = np.uint8(77)

# Instantiate the named tuple
# Since all fields have defaults, no arguments are needed
Label = BrainLabels()

SAS_LABEL_OFFSET = 10000
SPINAL_ID = 99


def fs_aparc_to_sas_marker(label):
    """Map a FreeSurfer aparc label to a brainmesh SAS marker (label + SAS_LABEL_OFFSET)."""
    return label + SAS_LABEL_OFFSET


def sas_marker_to_fs_aparc(label):
    """Recover the original FreeSurfer aparc label from a brainmesh SAS marker."""
    return label - SAS_LABEL_OFFSET


reverse_label_map = {
    getattr(Label, attr): attr
    for attr in dir(Label)
    if not attr.startswith('_') and isinstance(getattr(Label, attr), (int, np.integer))
}
reverse_label_map[0] = "BACKGROUND (0)"

VENTRICLE_LABELS = [
    Label.LEFT_LATERAL_VENTRICLE,
    Label.LEFT_INFERIOR_LATERAL_VENTRICLE,
    Label.RIGHT_LATERAL_VENTRICLE,
    Label.RIGHT_INFERIOR_LATERAL_VENTRICLE,
    Label.THIRD_VENTRICLE,
    Label.FOURTH_VENTRICLE,
    Label.RIGHT_CHOROID_PLEXUS,
    Label.LEFT_CHOROID_PLEXUS,
]

CSF_LABELS = VENTRICLE_LABELS + [Label.CSF]


def is_csf_marker(markers):
    """True where a marker belongs to the CSF compartment.

    That is CSF itself, any ventricle or choroid plexus, or a SAS parcel
    (``> SAS_LABEL_OFFSET``).  ``UNCLASSIFIED`` (vessels in the SAS) and
    ``SPINAL_BUFFER`` are deliberately *not* part of it.
    """
    markers = np.asarray(markers)
    return np.isin(markers, CSF_LABELS) | (markers > SAS_LABEL_OFFSET)


WM_LABELS = [Label.LEFT_CEREBRAL_WHITE_MATTER, 
             Label.RIGHT_CEREBRAL_WHITE_MATTER]
GM_LABELS = [Label.LEFT_CEREBRAL_CORTEX, 
             Label.RIGHT_CEREBRAL_CORTEX]

WM_CEREBELLUM_LABELS = [Label.LEFT_CEREBELLUM_WHITE_MATTER, 
                        Label.RIGHT_CEREBELLUM_WHITE_MATTER]
GM_CEREBELLUM_LABELS = [Label.LEFT_CEREBELLUM_CORTEX, 
                        Label.RIGHT_CEREBELLUM_CORTEX]

TISSUE_LABELS = list(set(Label._asdict().values()) - set(VENTRICLE_LABELS + [Label.CSF]))

# groups for csf facet regions
# ── DK40 aparc parcel → lobe groupings (FreeSurfer label values, pre-SAS-offset) ──
# LH labels 1001-1035; RH = LH + 1000 (2001-2035); SAS marker = FS label + SAS_LABEL_OFFSET
_LH_FRONTAL_FS   = [1003, 1012, 1018, 1019, 1020, 1027, 1028]
_LH_PARIETAL_FS  = [1005, 1008, 1017, 1022, 1024, 1025, 1029, 1031]
_LH_TEMPORAL_FS  = [1007, 1009, 1015, 1016, 1030]
_LH_OCCIPITAL_FS = [1011, 1013, 1021]

def _sas_lh(fs_set): return list(l + SAS_LABEL_OFFSET for l in fs_set)
def _sas_rh(fs_set): return list(l + 1000 + SAS_LABEL_OFFSET for l in fs_set)

region_dict = dict(
LEFT_FRONTAL_LOBE   = _sas_lh(_LH_FRONTAL_FS),
RIGHT_FRONTAL_LOBE   = _sas_rh(_LH_FRONTAL_FS),
LEFT_PARIETAL_LOBE  = _sas_lh(_LH_PARIETAL_FS),
RIGHT_PARIETAL_LOBE  = _sas_rh(_LH_PARIETAL_FS),
LEFT_TEMPORAL_LOBE  = _sas_lh(_LH_TEMPORAL_FS),
RIGHT_TEMPORAL_LOBE  = _sas_rh(_LH_TEMPORAL_FS),
LEFT_OCCIPITAL_LOBE = _sas_lh(_LH_OCCIPITAL_FS),
RIGHT_OCCIPITAL_LOBE = _sas_rh(_LH_OCCIPITAL_FS),
ANTERIOR_SKULL_BASE = list(l + SAS_LABEL_OFFSET for l in [28, 60, 1014, 2014, 1006, 2006]),
INFRATENTORIAL = list(l + SAS_LABEL_OFFSET for l in [6,7,8, 45,46,47, 16]),
)

