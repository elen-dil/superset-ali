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
import socket
import threading
from unittest.mock import Mock, patch

import paramiko
import pytest
import sshtunnel
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import (
    generate_private_key as generate_rsa_key,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from paramiko import (
    ECDSAKey,
    Ed25519Key,
    PasswordRequiredException,
    RSAKey,
    SSHException,
)

from superset.commands.database.ssh_tunnel.exceptions import (
    SSHTunnelHostKeyVerificationError,
)
from superset.extensions.ssh import _load_private_key, SSHManager, SSHManagerFactory


def _make_manager(strict: bool = False) -> SSHManager:
    """Build an ``SSHManager`` test instance with configurable strict checking."""
    app = Mock()
    app.config = {
        "SSH_TUNNEL_MAX_RETRIES": 2,
        "SSH_TUNNEL_LOCAL_BIND_ADDRESS": "127.0.0.1",
        "SSH_TUNNEL_TIMEOUT_SEC": 123.0,
        "SSH_TUNNEL_PACKET_TIMEOUT_SEC": 321.0,
        "SSH_TUNNEL_MANAGER_CLASS": "superset.extensions.ssh.SSHManager",
        "SSH_TUNNEL_STRICT_HOST_KEY_CHECKING": strict,
    }
    return SSHManager(app)


def _make_ssh_tunnel(private_key: str, private_key_password: str | None = None) -> Mock:
    ssh_tunnel = Mock()
    ssh_tunnel.server_address = "ssh.example.com"
    ssh_tunnel.server_port = 22
    ssh_tunnel.username = "tunneluser"
    ssh_tunnel.password = None
    ssh_tunnel.private_key = private_key
    ssh_tunnel.private_key_password = private_key_password
    # No expected host key: keeps create_tunnel on the key-parsing path without
    # triggering host-key verification (exercised separately below).
    ssh_tunnel.server_host_key = None
    return ssh_tunnel


def _authorized_key(key: paramiko.PKey) -> str:
    """Render a paramiko key in authorized-key (``"<type> <base64>"``) form."""
    return f"{key.get_name()} {key.get_base64()}"


def _ssh_tunnel(server_host_key: str | None) -> Mock:
    """Create a mocked SSH tunnel with server connection fields populated."""
    tunnel = Mock()
    tunnel.server_address = "ssh.example.com"
    tunnel.server_port = 22
    tunnel.server_host_key = server_host_key
    return tunnel


def test_ssh_tunnel_timeout_setting() -> None:
    app = Mock()
    app.config = {
        "SSH_TUNNEL_MAX_RETRIES": 2,
        "SSH_TUNNEL_LOCAL_BIND_ADDRESS": "test",
        "SSH_TUNNEL_TIMEOUT_SEC": 123.0,
        "SSH_TUNNEL_PACKET_TIMEOUT_SEC": 321.0,
        "SSH_TUNNEL_MANAGER_CLASS": "superset.extensions.ssh.SSHManager",
    }
    factory = SSHManagerFactory()
    factory.init_app(app)
    assert sshtunnel.TUNNEL_TIMEOUT == 123.0
    assert sshtunnel.SSH_TIMEOUT == 321.0


def _make_ed25519_pem() -> str:
    """Generate a fresh OpenSSH-format ed25519 private key PEM."""
    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.OpenSSH,
        encryption_algorithm=NoEncryption(),
    ).decode()


def test_create_tunnel_accepts_ed25519_private_key() -> None:
    """
    Regression for #24180: ed25519 SSH keys must be loadable for tunnel
    setup. The bug surfaces as ``unpack requires 4 bytes`` because the
    code unconditionally calls ``RSAKey.from_private_key`` regardless of
    key type, which mis-parses the ed25519 byte stream.

    The Superset UI accepts any key the user pastes, so the fix is to
    detect the key type (or use ``paramiko.PKey.from_private_key`` once
    paramiko exposes a polymorphic loader) rather than hard-coding RSA.

    This test exercises ``create_tunnel`` end-to-end with a freshly
    generated ed25519 key. It mocks ``sshtunnel.open_tunnel`` so the
    test does not actually open a network connection — only the key
    parsing path is exercised.
    """
    manager = _make_manager()
    ssh_tunnel = _make_ssh_tunnel(_make_ed25519_pem())

    with patch("superset.extensions.ssh.sshtunnel.open_tunnel") as mock_open:
        manager.create_tunnel(
            ssh_tunnel, "postgresql://user:pass@db.example.com:5432/x"
        )

    # Key-type-agnostic loader must produce a paramiko PKey usable as ssh_pkey.
    assert mock_open.called, "open_tunnel was never invoked — key parsing aborted"
    forwarded_pkey = mock_open.call_args.kwargs["ssh_pkey"]
    assert isinstance(forwarded_pkey, Ed25519Key)


def test_create_tunnel_accepts_rsa_private_key_unchanged() -> None:
    """
    Companion to test_create_tunnel_accepts_ed25519_private_key: pin the
    backward-compatible RSA path so a fix for ed25519 doesn't regress the
    historically-supported RSA case. Uses a freshly generated RSA key in
    OpenSSH format.
    """
    rsa_pem = (
        generate_rsa_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.OpenSSH,
            encryption_algorithm=NoEncryption(),
        )
        .decode()
    )

    manager = _make_manager()
    ssh_tunnel = _make_ssh_tunnel(rsa_pem)

    with patch("superset.extensions.ssh.sshtunnel.open_tunnel") as mock_open:
        manager.create_tunnel(
            ssh_tunnel, "postgresql://user:pass@db.example.com:5432/x"
        )

    assert mock_open.called, "open_tunnel was never invoked — RSA key parsing aborted"
    assert isinstance(mock_open.call_args.kwargs["ssh_pkey"], RSAKey)


def test_create_tunnel_accepts_ecdsa_private_key() -> None:
    """
    Companion to the ed25519 and RSA tests: verify ECDSAKey (the third type
    in _SSH_KEY_TYPES) is reachable by _load_private_key. Uses NIST P-256,
    the most common ECDSA curve in practice.
    """
    ecdsa_pem = (
        ec.generate_private_key(ec.SECP256R1())
        .private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.OpenSSH,
            encryption_algorithm=NoEncryption(),
        )
        .decode()
    )

    manager = _make_manager()
    ssh_tunnel = _make_ssh_tunnel(ecdsa_pem)

    with patch("superset.extensions.ssh.sshtunnel.open_tunnel") as mock_open:
        manager.create_tunnel(
            ssh_tunnel, "postgresql://user:pass@db.example.com:5432/x"
        )

    assert mock_open.called, "open_tunnel was never invoked — ECDSA key parsing aborted"
    assert isinstance(mock_open.call_args.kwargs["ssh_pkey"], ECDSAKey)


def test_create_tunnel_passphrase_protected_key_without_password() -> None:
    """
    A passphrase-protected key supplied without a passphrase must surface as
    ``PasswordRequiredException`` (an actionable "key requires passphrase"
    signal) rather than being absorbed by the per-type loop and reported as a
    generic "Unable to parse" error.
    """
    encrypted_pem = (
        Ed25519PrivateKey.generate()
        .private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.OpenSSH,
            encryption_algorithm=BestAvailableEncryption(b"correct horse"),
        )
        .decode()
    )

    manager = _make_manager()
    ssh_tunnel = _make_ssh_tunnel(encrypted_pem, private_key_password=None)

    with patch("superset.extensions.ssh.sshtunnel.open_tunnel") as mock_open:
        with pytest.raises(PasswordRequiredException):
            manager.create_tunnel(
                ssh_tunnel, "postgresql://user:pass@db.example.com:5432/x"
            )

    assert not mock_open.called


def test_create_tunnel_invalid_key_raises_combined_error() -> None:
    """
    When a key parses as none of the supported types, ``_load_private_key``
    raises ``SSHException`` whose message lists every type that was attempted,
    so the failure clearly communicates that all loaders were tried.
    """
    manager = _make_manager()
    ssh_tunnel = _make_ssh_tunnel("not a valid private key")

    with patch("superset.extensions.ssh.sshtunnel.open_tunnel") as mock_open:
        with pytest.raises(SSHException) as exc_info:
            manager.create_tunnel(
                ssh_tunnel, "postgresql://user:pass@db.example.com:5432/x"
            )

    message = str(exc_info.value)
    assert "Ed25519Key" in message
    assert "ECDSAKey" in message
    assert "RSAKey" in message
    assert not mock_open.called


@patch("superset.extensions.ssh.socket.create_connection")
@patch("superset.extensions.ssh.paramiko.Transport")
def test_verify_host_key_match(
    mock_transport_cls: Mock, mock_create_connection: Mock
) -> None:
    """The server presents the same key we expect: verification passes."""
    server_key = paramiko.RSAKey.generate(2048)
    manager = _make_manager(strict=False)
    tunnel = _ssh_tunnel(_authorized_key(server_key))

    transport = mock_transport_cls.return_value
    transport.get_remote_server_key.return_value = server_key

    result = manager._verify_host_key(tunnel)  # should not raise

    # The TCP connect is bounded by an explicit timeout, and the resulting
    # socket is handed to Transport.
    mock_create_connection.assert_called_once_with(
        ("ssh.example.com", 22), timeout=321.0
    )
    mock_transport_cls.assert_called_once_with(mock_create_connection.return_value)
    transport.start_client.assert_called_once()
    transport.close.assert_called_once()
    # The parsed expected key is returned so the caller can pin it on the tunnel.
    assert result == server_key


@patch("superset.extensions.ssh.socket.create_connection")
@patch("superset.extensions.ssh.paramiko.Transport")
def test_verify_host_key_mismatch_raises(
    mock_transport_cls: Mock, mock_create_connection: Mock
) -> None:
    """The server presents a different key than expected: verification fails."""
    expected_key = paramiko.RSAKey.generate(2048)
    presented_key = paramiko.RSAKey.generate(2048)
    manager = _make_manager(strict=False)
    tunnel = _ssh_tunnel(_authorized_key(expected_key))

    transport = mock_transport_cls.return_value
    transport.get_remote_server_key.return_value = presented_key

    with pytest.raises(SSHTunnelHostKeyVerificationError):
        manager._verify_host_key(tunnel)

    mock_create_connection.assert_called_once()
    transport.close.assert_called_once()


@patch("superset.extensions.ssh.socket.create_connection")
def test_verify_host_key_connect_failure_raises(
    mock_create_connection: Mock,
) -> None:
    """A bounded TCP connect failure surfaces as a host-key verification error."""
    manager = _make_manager(strict=False)
    server_key = paramiko.RSAKey.generate(2048)
    tunnel = _ssh_tunnel(_authorized_key(server_key))

    mock_create_connection.side_effect = OSError("connection refused")

    with pytest.raises(SSHTunnelHostKeyVerificationError):
        manager._verify_host_key(tunnel)


@patch("superset.extensions.ssh.paramiko.Transport")
def test_verify_host_key_unset_non_strict_skips(mock_transport_cls: Mock) -> None:
    """Back-compat: no expected key + strict checking off => no verification at all."""
    manager = _make_manager(strict=False)
    tunnel = _ssh_tunnel(None)

    assert manager._verify_host_key(tunnel) is None  # should not raise

    mock_transport_cls.assert_not_called()


@patch("superset.extensions.ssh.paramiko.Transport")
def test_verify_host_key_unset_strict_raises(mock_transport_cls: Mock) -> None:
    """Fail-closed: no expected key + strict checking on => reject."""
    manager = _make_manager(strict=True)
    tunnel = _ssh_tunnel(None)

    with pytest.raises(SSHTunnelHostKeyVerificationError):
        manager._verify_host_key(tunnel)

    mock_transport_cls.assert_not_called()


@patch("superset.extensions.ssh.socket.create_connection")
@patch("superset.extensions.ssh.paramiko.Transport")
def test_verify_host_key_match_ignores_comment_and_whitespace(
    mock_transport_cls: Mock,
    mock_create_connection: Mock,
) -> None:
    # The stored key may carry a trailing comment and extra whitespace.
    server_key = paramiko.RSAKey.generate(2048)
    manager = _make_manager(strict=False)
    stored = f"  {_authorized_key(server_key)} user@host  "
    tunnel = _ssh_tunnel(stored)

    transport = mock_transport_cls.return_value
    transport.get_remote_server_key.return_value = server_key

    manager._verify_host_key(tunnel)  # should not raise

    # Whitespace/comment stripping must not short-circuit verification: the
    # bounded TCP connect and Transport handshake still run as in the plain
    # match case.
    mock_create_connection.assert_called_once_with(
        ("ssh.example.com", 22), timeout=321.0
    )
    mock_transport_cls.assert_called_once_with(mock_create_connection.return_value)
    transport.start_client.assert_called_once()
    transport.close.assert_called_once()


def test_verify_host_key_invalid_expected_raises() -> None:
    # A malformed expected key is rejected before any network connection.
    manager = _make_manager(strict=False)
    tunnel = _ssh_tunnel("not-a-valid-key")

    with pytest.raises(SSHTunnelHostKeyVerificationError):
        manager._verify_host_key(tunnel)


def test_verify_host_key_unknown_key_type_raises() -> None:
    """An unsupported key type is wrapped in the verification error, not leaked."""
    manager = _make_manager(strict=False)
    server_key = paramiko.RSAKey.generate(2048)
    tunnel = _ssh_tunnel(f"ssh-bogus {server_key.get_base64()}")

    with pytest.raises(SSHTunnelHostKeyVerificationError):
        manager._verify_host_key(tunnel)


@patch("superset.extensions.ssh.sshtunnel.open_tunnel")
@patch("superset.extensions.ssh.socket.create_connection")
@patch("superset.extensions.ssh.paramiko.Transport")
def test_create_tunnel_pins_verified_host_key(
    mock_transport_cls: Mock,
    mock_create_connection: Mock,
    mock_open_tunnel: Mock,
) -> None:
    """A verified expected key is also pinned on the tunnel's own connection.

    When an expected host key is configured and verified, it is also pinned on the
    tunnel's own connection (``ssh_host_key``) so paramiko verifies the host that
    actually carries traffic on the same transport — closing the probe-vs-tunnel
    TOCTOU gap rather than trusting only the pre-flight probe.
    """
    server_key = paramiko.RSAKey.generate(2048)
    manager = _make_manager(strict=False)
    tunnel = _ssh_tunnel(_authorized_key(server_key))
    tunnel.username = "user"
    tunnel.password = None
    tunnel.private_key = None

    mock_transport_cls.return_value.get_remote_server_key.return_value = server_key

    manager.create_tunnel(tunnel, "postgresql://u:p@db:5432/ex")

    _, kwargs = mock_open_tunnel.call_args
    assert kwargs["ssh_host_key"] == server_key


@patch("superset.extensions.ssh.sshtunnel.open_tunnel")
def test_create_tunnel_without_host_key_does_not_pin(mock_open_tunnel: Mock) -> None:
    # No expected key configured (non-strict): nothing is pinned, preserving the
    # prior behavior.
    manager = _make_manager(strict=False)
    tunnel = _ssh_tunnel(None)
    tunnel.username = "user"
    tunnel.password = None
    tunnel.private_key = None

    manager.create_tunnel(tunnel, "postgresql://u:p@db:5432/ex")

    _, kwargs = mock_open_tunnel.call_args
    assert "ssh_host_key" not in kwargs


def test_ssh_tunnel_schema_round_trips_server_host_key() -> None:
    """The schema accepts and preserves the public host key field."""
    from superset.databases.schemas import DatabaseSSHTunnel

    server_key = paramiko.RSAKey.generate(2048)
    authorized = _authorized_key(server_key)
    payload = {
        "server_address": "ssh.example.com",
        "server_port": 22,
        "username": "user",
        "password": "secret",
        "server_host_key": authorized,
    }
    loaded = DatabaseSSHTunnel().load(payload)
    assert loaded["server_host_key"] == authorized


def test_paramiko_rejects_sha1_rsa_signatures() -> None:
    """
    CVE-2026-44405 / GHSA-r374-rxx8-8654: paramiko must not sign, verify or
    negotiate RSA with SHA-1 (``ssh-rsa`` as a signature algorithm). RSA keys
    themselves remain usable via the SHA-2 ``rsa-sha2-*`` algorithms.
    """
    key = RSAKey.generate(2048)

    assert "ssh-rsa" not in RSAKey.HASHES
    with pytest.raises(KeyError):
        key.sign_ssh_data(b"payload", algorithm="ssh-rsa")
    assert "ssh-rsa" not in paramiko.Transport._preferred_keys  # type: ignore[attr-defined]
    assert "ssh-rsa" not in paramiko.Transport._preferred_pubkeys  # type: ignore[attr-defined]

    msg = key.sign_ssh_data(b"payload", algorithm="rsa-sha2-256")
    msg.rewind()
    assert key.verify_ssh_sig(b"payload", msg)


class _EchoSSHServer(paramiko.ServerInterface):
    """In-process SSH server accepting one public key and echoing forwarded data."""

    def __init__(self, client_key: paramiko.PKey) -> None:
        self.client_key = client_key
        self.destinations: list[tuple[str, int]] = []

    def get_allowed_auths(self, username: str) -> str:
        return "publickey"

    def check_auth_publickey(self, username: str, key: paramiko.PKey) -> int:
        if username == "tunneluser" and key == self.client_key:
            return paramiko.common.AUTH_SUCCESSFUL
        return paramiko.common.AUTH_FAILED

    def check_channel_direct_tcpip_request(
        self, chanid: int, origin: tuple[str, int], destination: tuple[str, int]
    ) -> int:
        self.destinations.append(destination)
        return paramiko.common.OPEN_SUCCEEDED


def _serve_ssh(
    listener: socket.socket,
    host_key: paramiko.PKey,
    server: _EchoSSHServer,
    transports: list[paramiko.Transport],
) -> None:
    while True:
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        transport = paramiko.Transport(conn)
        transport.add_server_key(host_key)
        transports.append(transport)
        try:
            transport.start_server(server=server)
        except (paramiko.SSHException, EOFError):
            continue
        threading.Thread(target=_echo_channels, args=(transport,), daemon=True).start()


def _echo_channels(transport: paramiko.Transport) -> None:
    while transport.is_active():
        chan = transport.accept(timeout=0.5)
        if chan is None:
            continue
        while data := chan.recv(1024):
            chan.sendall(data)
        chan.close()


def test_create_tunnel_end_to_end_with_rsa_keys() -> None:
    """
    SSH tunneling keeps working end to end with RSA host and client keys, using
    SHA-2 signatures only: the tunnel authenticates, pins the host key, and
    forwards traffic to the remote bind address.
    """
    host_key = RSAKey.generate(2048)
    client_key = generate_rsa_key(public_exponent=65537, key_size=2048)
    client_pem = client_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.OpenSSH,
        encryption_algorithm=NoEncryption(),
    ).decode()

    server = _EchoSSHServer(_load_private_key(client_pem, None))
    transports: list[paramiko.Transport] = []
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    threading.Thread(
        target=_serve_ssh,
        args=(listener, host_key, server, transports),
        daemon=True,
    ).start()

    app = Mock()
    app.config = {
        "SSH_TUNNEL_LOCAL_BIND_ADDRESS": "127.0.0.1",
        "SSH_TUNNEL_TIMEOUT_SEC": 10.0,
        "SSH_TUNNEL_PACKET_TIMEOUT_SEC": 1.0,
        "SSH_TUNNEL_STRICT_HOST_KEY_CHECKING": True,
    }
    manager = SSHManager(app)
    ssh_tunnel = _make_ssh_tunnel(client_pem)
    ssh_tunnel.server_address = "127.0.0.1"
    ssh_tunnel.server_port = listener.getsockname()[1]
    ssh_tunnel.server_host_key = _authorized_key(host_key)

    forwarder = manager.create_tunnel(
        ssh_tunnel, "postgresql://user:pass@db.example.com:5432/x"
    )
    try:
        with forwarder:
            assert forwarder._transport.host_key_type in RSAKey.HASHES
            assert forwarder._transport._agreed_pubkey_algorithm in RSAKey.HASHES
            with socket.create_connection(
                ("127.0.0.1", forwarder.local_bind_port), timeout=5
            ) as client:
                client.sendall(b"select 1")
                assert client.recv(1024) == b"select 1"
    finally:
        listener.close()
        for transport in transports:
            transport.close()

    assert server.destinations == [("db.example.com", 5432)]
