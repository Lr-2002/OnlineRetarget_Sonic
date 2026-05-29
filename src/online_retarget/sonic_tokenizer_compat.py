"""Compatibility patch for OnlineRetarget-added SONIC tokenizer terms."""

from __future__ import annotations

import builtins
import inspect
import sys
from types import ModuleType
from typing import Any
import warnings
import weakref


ONLINE_RETARGET_TOKENIZER_TERMS = (
    "soma_morphology",
    "soma_contact_phase",
    "root_pos_w_mf",
    "root_rot_w_mf",
)

_IMPORT_PATCHED = False
_ORIGINAL_IMPORT = builtins.__import__
_MODULE_NAME_TOKENS = ("gear_sonic", "isaaclab", "online_retarget")
_PATCHED_TOKENIZER_CFGS: weakref.WeakSet[type[Any]] = weakref.WeakSet()


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
        _patch_imported_modules(name, module, fromlist)
        return module

    builtins.__import__ = importing
    _IMPORT_PATCHED = True


def _patch_loaded_tokenizer_cfgs() -> None:
    seen: set[int] = set()
    for name, module in tuple(sys.modules.items()):
        if module is None or not _is_candidate_module_name(name):
            continue
        _patch_module_tokenizer_cfgs(module, seen)


def _patch_imported_modules(name: str, module: ModuleType, fromlist: Any) -> None:
    seen: set[int] = set()
    for candidate in _candidate_import_modules(name, module, fromlist):
        if candidate is not None:
            _patch_module_tokenizer_cfgs(candidate, seen)


def _candidate_import_modules(
    name: str,
    module: ModuleType,
    fromlist: Any,
) -> tuple[ModuleType | None, ...]:
    candidates: list[ModuleType | None] = []
    if _is_candidate_module_name(name):
        candidates.append(sys.modules.get(name))
    if _is_candidate_module_name(getattr(module, "__name__", "")):
        candidates.append(module)
    for item in fromlist or ():
        child = getattr(module, str(item), None)
        if isinstance(child, ModuleType) and _is_candidate_module_name(child.__name__):
            candidates.append(child)
    return tuple(candidates)


def _is_candidate_module_name(name: str) -> bool:
    return any(token in name for token in _MODULE_NAME_TOKENS)


def _patch_module_tokenizer_cfgs(module: ModuleType, seen: set[int]) -> None:
    for value in vars(module).values():
        _patch_tokenizer_cfgs_in_value(value, seen)


def _patch_tokenizer_cfgs_in_value(value: Any, seen: set[int]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
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
    if cls in _PATCHED_TOKENIZER_CFGS:
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
    _PATCHED_TOKENIZER_CFGS.add(cls)
