# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Regression tests for PyJWT CVE-2026-101918 and CVE-2026-102275."""

import base64
import hashlib
import hmac

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from flask import current_app
from jwt.algorithms import OKPAlgorithm

from superset.extensions import appbuilder
from superset.security.manager import SupersetSecurityManager

NESTING_DEPTH = 20_000


def _b64url(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _deeply_nested_hs256_token(secret: str) -> str:
    header = _b64url(b'{"alg": "HS256", "typ": "JWT"}')
    payload = _b64url(b"[" * NESTING_DEPTH + b"]" * NESTING_DEPTH)
    signing_input = header + b"." + payload
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return (signing_input + b"." + _b64url(signature)).decode()


def test_guest_token_with_deeply_nested_payload_raises_decode_error(
    app_context: None,
) -> None:
    """CVE-2026-101918: a recursive payload must surface as ``DecodeError``."""
    assert current_app.config["GUEST_TOKEN_JWT_ALGO"] == "HS256"  # noqa: S105
    sm = SupersetSecurityManager(appbuilder)
    token = _deeply_nested_hs256_token(current_app.config["GUEST_TOKEN_JWT_SECRET"])

    with pytest.raises(jwt.DecodeError):
        sm.parse_jwt_guest_token(token)


def test_unverified_decode_with_deeply_nested_payload_raises_decode_error() -> None:
    """CVE-2026-101918: the pre-verification decode path is also hardened."""
    token = _deeply_nested_hs256_token("irrelevant")

    with pytest.raises(jwt.DecodeError):
        jwt.decode(token, options={"verify_signature": False})


def _b64url_str(data: bytes) -> str:
    return _b64url(data).decode()


def test_okp_private_jwk_with_mismatched_public_key_is_rejected() -> None:
    """CVE-2026-102275: ``x`` must correspond to the private key ``d``."""
    private_key = Ed25519PrivateKey.generate()
    other_public_key = Ed25519PrivateKey.generate().public_key()
    jwk = {
        "kty": "OKP",
        "crv": "Ed25519",
        "d": _b64url_str(private_key.private_bytes_raw()),
        "x": _b64url_str(other_public_key.public_bytes_raw()),
    }

    with pytest.raises(jwt.InvalidKeyError):
        OKPAlgorithm.from_jwk(jwk)
