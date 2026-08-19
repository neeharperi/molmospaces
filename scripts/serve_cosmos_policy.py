#!/usr/bin/env python3
"""Serve Cosmos3-{Edge,Nano}-Policy-DROID for the eval harness.

The one behavioral patch this wrapper exists for: ``RobolabPolicyService.__init__`` calls
``OmniInference.create(setup_args)``, which unconditionally tries to load a text-guardrail model
(``Blocklist``, from the ``nvidia/Cosmos-Guardrail1`` checkpoint) whenever ``setup_args.guardrails``
is true -- the default, and not exposed as a flag anywhere in
``action_policy_server_robolab.py``'s own CLI surface. That checkpoint is HF-gated and requires
approval, unlike the two DROID policy checkpoints themselves, so the server 500s at startup with
"Access denied. This repository requires approval." before ever reaching a real request.

Content-safety filtering on generated video is irrelevant to a closed-loop robot-manipulation
benchmark run with no public output, so this wrapper disables it rather than waiting on HF access
approval. Set ``COSMOS_POLICY_GUARDRAILS=1`` to re-enable once/if access to
``nvidia/Cosmos-Guardrail1`` is granted. Ported verbatim from a sibling project's own working
wrapper for this exact server -- see docs/eval_reproduction.md's Cosmos-Policy section.
"""

import os

import cosmos_framework.scripts.action_policy_server_robolab as server_module

_GUARDRAILS = os.getenv("COSMOS_POLICY_GUARDRAILS", "0").lower() in ("1", "true", "yes")


def _patch_guardrails() -> None:
    original_build_setup_args = server_module.RobolabPolicyService._build_setup_args

    def _build_setup_args(self, args):
        setup_args = original_build_setup_args(self, args)
        setup_args.guardrails = _GUARDRAILS
        return setup_args

    server_module.RobolabPolicyService._build_setup_args = _build_setup_args


def main() -> None:
    _patch_guardrails()
    print(f"[serve_cosmos_policy] guardrails: {'on' if _GUARDRAILS else 'off'}", flush=True)
    server_module.main()


if __name__ == "__main__":
    main()
