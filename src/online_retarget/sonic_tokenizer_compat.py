"""Compatibility patch for OnlineRetarget-added SONIC tokenizer terms."""

from __future__ import annotations

import builtins
import inspect
import sys
from types import ModuleType
from typing import Any


ONLINE_RETARGET_TOKENIZER_TERMS = (
    "soma_morphology",
    "soma_contact_phase",
    "root_pos_w_mf",
    "root_rot_w_mf",
)

_IMPORT_PATCHED = False
_ORIGINAL_IMPORT = builtins.__import__


def install_tokenizer_cfg_compat() -> None:
    """Allow structured upstream ``TokenizerCfg`` classes to carry local terms.

    Some SONIC/IsaacLab revisions instantiate ``manager_env.observations.tokenizer``
    through a structured ``TokenizerCfg`` class.  Hydra can compose the extra
    OnlineRetarget observation terms, but the class constructor may reject them
    as unknown keyword arguments before training reaches the kin-only model path.
    This shim keeps the upstream class behavior and stores only the local terms
    that the constructor does not already accept.
    """

    global _IMPORT_PATCHED
    _patch_loaded_tokenizer_cfgs()
    if _IMPORT_PATCHED:
        return

    def importing(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> ModuleType:
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        _patch_loaded_tokenizer_cfgs()
        return module

    builtins.__import__ = importing
    _IMPORT_PATCHED = True


def _patch_loaded_tokenizer_cfgs() -> None:
    seen: set[int] = set()
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        for value in vars(module).values():
            _patch_tokenizer_cfgs_in_value(value, seen)


def _patch_tokenizer_cfgs_in_value(value: Any, seen: set[int]) -> None:
    if not inspect.isclass(value):
        return
    ident = id(value)
    if ident in seen:
        return
    seen.add(ident)

    if value.__name__ == "TokenizerCfg":
        _patch_tokenizer_cfg(value)
    for child in vars(value).values():
        if inspect.isclass(child):
            _patch_tokenizer_cfgs_in_value(child, seen)


def _patch_tokenizer_cfg(cls: type[Any]) -> None:
    if getattr(cls, "_online_retarget_tokenizer_compat", False):
        return

    original_init = getattr(cls, "__init__", None)
    if original_init is None:
        return

    signature = inspect.signature(original_init)
    params = signature.parameters
    accepts_var_kwargs = any(param.kind is param.VAR_KEYWORD for param in params.values())
    accepted = {
        name
        for name, param in params.items()
        if name != "self"
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        local_terms = {}
        if not accepts_var_kwargs:
            for term in ONLINE_RETARGET_TOKENIZER_TERMS:
                if term in kwargs and term not in accepted:
                    local_terms[term] = kwargs.pop(term)

        original_init(self, *args, **kwargs)

        for term, value in local_terms.items():
            try:
                setattr(self, term, value)
            except AttributeError:
                object.__setattr__(self, term, value)

    cls.__init__ = patched_init
    cls._online_retarget_tokenizer_compat = True
