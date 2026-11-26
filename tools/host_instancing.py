"""Explicit bounded host attribute-divisor instancing contract."""

from tools.mgcdriver import ProgramFormat


HOST_INSTANCING_ABI_VERSION = 3
HOST_INSTANCE_ID_WORD = 65535
HOST_INSTANCING_CONTRACT = {
    "version": 1,
    "max_instances": 16,
    "max_vertices": 4096,
    "indexing": "base-instance-plus-floor-local-instance-over-divisor",
    "shader_abi_version": HOST_INSTANCING_ABI_VERSION,
    "instance_id_word": HOST_INSTANCE_ID_WORD,
}


def shader_instancing_contract(uses_instance_id=False):
    if type(uses_instance_id) is not bool:
        raise ValueError("instance-ID use must be explicitly boolean")
    return {"version": 1, "max_instances": 16, "instance_id_word": HOST_INSTANCE_ID_WORD,
            "instance_id": "local-zero-based", "uses_instance_id": uses_instance_id}


def validate_shader_instancing(source):
    if source.get("version") != 4 and ("host_features" in source or source.get("shader_abi_version") == 4):
        raise ValueError("host features require explicit layout4/ABI4")
    if source.get("version") == 4:
        from tools.host_features import validate_shader_features

        return validate_shader_features(source)
    if source.get("version") != 3:
        if "host_instancing" in source or source.get("shader_abi_version") == HOST_INSTANCING_ABI_VERSION:
            raise ValueError("host shader instancing requires layout3 and ABI3")
        return None
    if (source.get("supported") is not True or source.get("program_format") != ProgramFormat.LEGACY.value or
            type(source.get("shader_abi_version")) is not int or
            source["shader_abi_version"] != HOST_INSTANCING_ABI_VERSION or
            source.get("address_units") != "words" or source.get("stage") not in ("vertex", "fragment")):
        raise ValueError("shader instancing requires supported legacy layout3/ABI3")
    contract = source.get("host_instancing")
    if type(contract) is not dict:
        raise ValueError("shader instancing requires an exact contract")
    expected = shader_instancing_contract(contract.get("uses_instance_id"))
    if (set(contract) != set(expected) or any(type(contract[key]) is not type(value) or contract[key] != value
                                            for key, value in expected.items()) or
            (source["stage"] != "vertex" and contract["uses_instance_id"])):
        raise ValueError("unsupported shader instancing contract")
    return contract


def validate_instancing(artifact):
    if type(artifact) is not dict:
        raise ValueError("host instancing requires a captured draw object")
    schema = artifact.get("schema")
    if schema != "mgc-host-draw-v4" and ("host_features" in artifact or artifact.get("shader_abi_version") == 4):
        raise ValueError("host features require version4 captured resources")
    if schema == "mgc-host-draw-v4":
        from tools.host_features import validate_draw_features

        return validate_draw_features(artifact)
    if schema in ("mgc-host-draw-v1", "mgc-host-draw-v2"):
        if "host_instancing" in artifact or artifact.get("shader_abi_version") == HOST_INSTANCING_ABI_VERSION:
            raise ValueError("host instancing requires version3 captured resources")
        return False
    if schema != "mgc-host-draw-v3":
        raise ValueError("unsupported Gallium capture schema")
    contract = artifact.get("host_instancing")
    if (type(contract) is not dict or set(contract) != set(HOST_INSTANCING_CONTRACT) or
            any(type(contract[key]) is not type(value) or contract[key] != value
                for key, value in HOST_INSTANCING_CONTRACT.items())):
        raise ValueError("unsupported exact host instancing contract")
    if (artifact.get("program_format") != ProgramFormat.LEGACY.value or
            type(artifact.get("shader_abi_version")) is not int or
            artifact["shader_abi_version"] != HOST_INSTANCING_ABI_VERSION):
        raise ValueError("host instancing is supported only by the legacy host path")
    return True
