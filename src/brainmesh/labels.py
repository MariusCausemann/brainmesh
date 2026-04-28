import numpy as np
from collections import namedtuple

_labels_dict = {
    'LEFT_CEREBRAL_WHITE_MATTER': np.uint8(2),
    'LEFT_CEREBRAL_CORTEX': np.uint8(3),
    'LEFT_LATERAL_VENTRICLE': np.uint8(4),
    'LEFT_INFERIOR_LATERAL_VENTRICLE': np.uint8(5),
    'LEFT_CEREBELLUM_WHITE_MATTER': np.uint8(7),
    'LEFT_CEREBELLUM_CORTEX': np.uint8(8),
    'LEFT_THALAMUS': np.uint8(10),
    'LEFT_CAUDATE': np.uint8(11),
    'LEFT_PUTAMEN': np.uint8(12),
    'LEFT_PALLIDUM': np.uint8(13),
    'THIRD_VENTRICLE': np.uint8(14),
    'FOURTH_VENTRICLE': np.uint8(15),
    'BRAIN_STEM': np.uint8(16),
    'LEFT_HIPPOCAMPUS': np.uint8(17),
    'LEFT_AMYGDALA': np.uint8(18),
    'CSF': np.uint8(24),
    'LEFT_ACCUMBENS_AREA': np.uint8(26),
    'LEFT_VENTRAL_DC': np.uint8(28),
    'LEFT_CHOROID_PLEXUS': np.uint8(31),
    'RIGHT_CEREBRAL_WHITE_MATTER': np.uint8(41),
    'RIGHT_CEREBRAL_CORTEX': np.uint8(42),
    'RIGHT_LATERAL_VENTRICLE': np.uint8(43),
    'RIGHT_INFERIOR_LATERAL_VENTRICLE': np.uint8(44),
    'RIGHT_CEREBELLUM_WHITE_MATTER': np.uint8(46),
    'RIGHT_CEREBELLUM_CORTEX': np.uint8(47),
    'RIGHT_THALAMUS': np.uint8(49),
    'RIGHT_CAUDATE': np.uint8(50),
    'RIGHT_PUTAMEN': np.uint8(51),
    'RIGHT_PALLIDUM': np.uint8(52),
    'RIGHT_HIPPOCAMPUS': np.uint8(53),
    'RIGHT_AMYGDALA': np.uint8(54),
    'RIGHT_ACCUMBENS_AREA': np.uint8(58),
    'RIGHT_VENTRAL_DC': np.uint8(60),
    'RIGHT_CHOROID_PLEXUS': np.uint8(63),
    'FALX': np.uint8(70),
    'TENTORIUM': np.uint8(71),
    'WM_HYPOINTENSITIES': np.uint8(77),
}

BrainLabels = namedtuple('BrainLabels', _labels_dict.keys())
Label = BrainLabels(**_labels_dict)

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

WM_LABELS = [Label.LEFT_CEREBRAL_WHITE_MATTER, 
             Label.RIGHT_CEREBRAL_WHITE_MATTER]
GM_LABELS = [Label.LEFT_CEREBRAL_CORTEX, 
             Label.RIGHT_CEREBRAL_CORTEX]

WM_CEREBELLUM_LABELS = [Label.LEFT_CEREBELLUM_WHITE_MATTER, 
                        Label.RIGHT_CEREBELLUM_WHITE_MATTER]
GM_CEREBELLUM_LABELS = [Label.LEFT_CEREBELLUM_CORTEX, 
                        Label.RIGHT_CEREBELLUM_CORTEX]

