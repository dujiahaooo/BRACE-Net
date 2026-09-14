"""Named, frozen model configurations reported in the paper."""

from models.bcresnet import BCResNet
from models.bracenet import BCDualNet
from models.tfdbp_style import TFDBPStyleTopologyControl


MODEL_NAMES = (
    "bc_resnet6",
    "bc_resnet8",
    "brace_lite",
    "brace_full",
    "widened_bc_only",
    "deep_se",
    "deep_se_exres",
    "deep_tace",
    "serial_local_tace",
    "serial_tace_local",
    "tfdbp_style_topology_control",
    "bc_resnet_compute76",
)


def _full_config(num_classes, **changes):
    config = dict(
        base_c=40,
        num_classes=num_classes,
        use_dual=True,
        use_tfca=True,
        use_ssn=True,
        use_extra_res=True,
        dual_stages=(2, 3),
        topology="parallel",
    )
    config.update(changes)
    return config


def create_model(name: str, num_classes: int = 35, **overrides):
    if name == "bc_resnet6":
        config = dict(base_c=48, num_classes=num_classes)
        config.update(overrides)
        return BCResNet(**config)
    if name == "bc_resnet8":
        config = dict(base_c=64, num_classes=num_classes)
        config.update(overrides)
        return BCResNet(**config)
    if name == "widened_bc_only":
        config = dict(base_c=60, num_classes=num_classes, use_dual=False)
        config.update(overrides)
        return BCDualNet(**config)
    if name == "bc_resnet_compute76":
        config = dict(base_c=76, num_classes=num_classes)
        config.update(overrides)
        return BCResNet(**config)
    if name == "brace_lite":
        config = dict(
            base_c=28,
            num_classes=num_classes,
            use_dual=True,
            use_tfca=True,
            use_ssn=True,
            use_extra_res=True,
            dual_stages=(0, 1, 2, 3),
            topology="parallel",
        )
        config.update(overrides)
        return BCDualNet(**config)
    variants = {
        "brace_full": {},
        "deep_se": {"use_tfca": False, "use_extra_res": False},
        "deep_se_exres": {"use_tfca": False, "use_extra_res": True},
        "deep_tace": {"use_tfca": True, "use_extra_res": False},
        "serial_local_tace": {"topology": "local_then_tace"},
        "serial_tace_local": {"topology": "tace_then_local"},
    }
    if name in variants:
        config = _full_config(num_classes, **variants[name])
        config.update(overrides)
        return BCDualNet(**config)
    if name == "tfdbp_style_topology_control":
        config = dict(base_c=40, num_classes=num_classes)
        config.update(overrides)
        return TFDBPStyleTopologyControl(**config)
    raise ValueError(f"Unknown model {name!r}; choices: {', '.join(MODEL_NAMES)}")
